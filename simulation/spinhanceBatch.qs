/**
 * batch_simulate.qs  —  SpinHance batch spin simulation
 * -------------------------------------------------------
 * MNova 16 JavaScript (.qs) batch script.
 *
 * INSTALL (once):
 *   cp simulation/batch_simulate.qs \
 *      /Applications/MestReNova.app/Contents/Resources/scripts/
 *
 * INVOKE:
 *   # 1. Write config (done automatically by run_batch.py):
 *   python -c "
 *   import json; from pathlib import Path
 *   Path.home().joinpath('.spinhance_batch_config.json').write_text(
 *       json.dumps({'xml_dir': '/path/to/xmls', 'out_dir': '/path/to/out'}))
 *   "
 *   # 2. Run MNova:
 *   /Applications/MestReNova.app/Contents/MacOS/MestReNova \
 *       --sf "spinhanceBatch()"
 *
 * API REFERENCES (confirmed from Mnova 15/16 docs + bundled examples):
 *   serialization.open(path)          — open a file, becomes activeDocument
 *   Application.nmr.activeSpectrum()  — get current NMR spectrum
 *   spec.count()                       — number of spectral points
 *   spec.real(i)                       — real intensity at point i
 *   mainWindow.activeDocument.close(false) — close without saving
 *   new NMRSpectrum(rawSpec)           — wrap raw spectrum object
 *   new File(path) + new TextStream(f) — write to file
 *   Dir.home()                         — user home directory string
 *   dir.entryList(mask, flags)         — list files in directory
 *   dir.exists                         — boolean property (not method)
 *   dir.mkpath(path)                   — create directory tree
 */

/*globals Application, mainWindow, serialization, NMRSpectrum,
          File, TextStream, Dir, JSON, print*/
/*jslint plusplus: true, indent: 4*/

// ── Main entry point ──────────────────────────────────────────────────────────

function spinhanceBatch() {
    "use strict";

    // ── 1. Read config JSON ───────────────────────────────────────────────────
    var configPath = Dir.home() + "/.spinhance_batch_config.json";
    var configFile = new File(configPath);
    if (!configFile.exists) {
        print("ERROR: config file not found: " + configPath);
        print("Write it via run_batch.py before invoking MNova.");
        return;
    }
    configFile.open(File.ReadOnly);
    var ts = new TextStream(configFile);
    var raw = ts.readAll();
    configFile.close();

    var config = JSON.parse(raw);
    var xmlDir  = config.xml_dir;
    var outDir  = config.out_dir;

    if (!xmlDir || !outDir) {
        print("ERROR: config must contain xml_dir and out_dir");
        return;
    }

    // ── 2. List XML files ─────────────────────────────────────────────────────
    var dir = new Dir(xmlDir);
    if (!dir.exists) {
        print("ERROR: xml_dir does not exist: " + xmlDir);
        return;
    }

    // entryList(nameFilter, filterFlags) — confirmed from Dir.qs + example scripts
    var files = dir.entryList("*.xml", Dir.Files);

    if (!files || files.length === 0) {
        print("No XML files found in: " + xmlDir);
        return;
    }
    print("Found " + files.length + " XML files. Starting batch simulation...");

    // Ensure output directory exists
    var outDirObj = new Dir(outDir);
    if (!outDirObj.exists) {
        outDirObj.mkpath(outDir);
    }

    // ── 3. Batch loop ─────────────────────────────────────────────────────────
    var succeeded = 0;
    var failed    = 0;

    for (var i = 0; i < files.length; i++) {
        var fname   = files[i];
        var xmlPath = xmlDir + "/" + fname;
        var stem    = fname.replace(/\.xml$/i, "");
        var outPath = outDir + "/" + stem + ".txt";

        print("[" + (i + 1) + "/" + files.length + "] " + fname + " ...");

        var ok = _processOne(xmlPath, outPath);
        if (ok) {
            succeeded++;
        } else {
            failed++;
            print("  FAILED: " + fname);
        }
    }

    print("");
    print("Done.  Succeeded: " + succeeded + "  Failed: " + failed);

    // ── 4. Quit MNova ─────────────────────────────────────────────────────────
    Application.mainWindow.close();
}


// ── Process a single XML file ─────────────────────────────────────────────────

function _processOne(xmlPath, outPath) {
    "use strict";

    // serialization.open() opens the file and makes it the activeDocument.
    // For mnova-spinsim XMLs, MNova runs the QM spin simulation synchronously.
    serialization.open(xmlPath);

    // Get the simulated spectrum via Application.nmr (confirmed in Mnova docs)
    var rawSpec = Application.nmr.activeSpectrum();
    if (!rawSpec || !rawSpec.isValid()) {
        print("  WARN: no valid spectrum after opening " + xmlPath);
        mainWindow.activeDocument.close(false);
        return false;
    }

    var spec    = new NMRSpectrum(rawSpec);
    var nPoints = spec.count();

    if (nPoints === 0) {
        print("  WARN: spectrum has 0 points for " + xmlPath);
        mainWindow.activeDocument.close(false);
        return false;
    }

    // Write one intensity value per line
    var f = new File(outPath);
    if (!f.open(File.WriteOnly | File.Text)) {
        print("  ERROR: cannot write to " + outPath);
        mainWindow.activeDocument.close(false);
        return false;
    }
    var stream = new TextStream(f);
    for (var j = 0; j < nPoints; j++) {
        stream.writeln(spec.real(j));
    }
    f.close();

    mainWindow.activeDocument.close(false);
    print("  ok (" + nPoints + " pts)");
    return true;
}

// Auto-execute when this file is loaded directly via --sf /path/to/spinhanceBatch.qs
spinhanceBatch();
