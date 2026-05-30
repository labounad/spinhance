//<GUI menuname="SpinHance Batch" shortcut="Ctrl+9" tooltip="Run SpinHance batch spin simulation"/>
/**
 * spinhanceBatch.qs  —  SpinHance batch spin simulation
 * -------------------------------------------------------
 * MNova 16 JavaScript (.qs) batch script.
 *
 * THREE RULES that make -sf work (these were the cause of the earlier failures):
 *   1. File name MUST equal the function name: this file is spinhanceBatch.qs
 *      and it defines function spinhanceBatch(...).
 *   2. The folder containing this file MUST be registered in
 *      Edit > Preferences > Scripting > Directories (do this once via the GUI,
 *      then restart MNova). Mnova only resolves -sf names from registered dirs.
 *   3. NO top-level auto-executing call. A bare spinhanceBatch() at file scope
 *      runs at startup before the NMR API exists and crashes MNova. -sf calls
 *      the function for you.
 *
 * INSTALL (once):
 *   - Put this file in a folder, e.g. ~/mnova_scripts/spinhanceBatch.qs
 *   - GUI: Edit > Preferences > Scripting > Directories > add ~/mnova_scripts
 *   - Restart MNova.
 *
 * INVOKE (correct syntax — single dash, NO parentheses, args comma-separated):
 *   /Applications/MestReNova.app/Contents/MacOS/MestReNova \
 *       -sf spinhanceBatch,/path/to/xml_dir,/path/to/out_dir
 *
 *   You can also run with no args; it then falls back to the config JSON at
 *   ~/.spinhance_batch_config.json  ({"xml_dir": "...", "out_dir": "..."}).
 *
 *   NOTE: pass -nogui to suppress the main window if your MNova build supports
 *   it. If -nogui causes problems, run with the GUI visible — it still batches.
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

function spinhanceBatch(argXmlDir, argOutDir) {
    "use strict";

    var xmlDir;
    var outDir;

    // ── 1. Prefer command-line args (-sf spinhanceBatch,xmlDir,outDir) ────────
    if (argXmlDir && argOutDir) {
        xmlDir = String(argXmlDir);
        outDir = String(argOutDir);
    } else {
        // Fall back to config JSON written by run_batch.py
        var configPath = Dir.home() + "/.spinhance_batch_config.json";
        var configFile = new File(configPath);
        if (!configFile.exists) {
            print("ERROR: no args given and config file not found: " + configPath);
            return;
        }
        configFile.open(File.ReadOnly);
        var ts = new TextStream(configFile);
        var raw = ts.readAll();
        configFile.close();

        var config = JSON.parse(raw);
        xmlDir = config.xml_dir;
        outDir = config.out_dir;
    }

    if (!xmlDir || !outDir) {
        print("ERROR: xml_dir and out_dir must both be provided");
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

// NO top-level call here. A bare spinhanceBatch() at file scope runs when MNova
// loads the script at startup — before the NMR API is ready — and crashes MNova.
// -sf invokes spinhanceBatch() for you after the app is initialized.
