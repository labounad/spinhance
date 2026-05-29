/**
 * batch_simulate.qs  —  SpinHance batch spin simulation
 * -------------------------------------------------------
 * MNova 16 JavaScript script. Must be installed in the user scripts directory
 * so MNova auto-loads it, then invoked via --sf:
 *
 *   cp simulation/batch_simulate.qs \
 *      ~/Library/Application\ Support/Mestrelab\ Research\ S.L./MestReNova/scripts/
 *
 *   /Applications/MestReNova.app/Contents/MacOS/MestReNova \
 *       --nogui --sf "spinhanceBatch()"
 *
 * Config is passed via a JSON file written by run_batch.py:
 *   /tmp/spinhance_batch_config.json
 *   { "xml_dir": "/abs/path/to/xmls", "out_dir": "/abs/path/to/out" }
 *
 * For each XML in xml_dir this script:
 *   1. Opens the mnova-spinsim XML  (MNova auto-runs the simulation)
 *   2. Reads the intensity array via nmr.activeSpectrum() + spec.real(i)
 *   3. Writes one .txt file per spectrum (one float per line)
 *   4. Closes the document
 *
 * API references confirmed from MNova example scripts:
 *   toPositive.qs        — spec.count(), spec.real(i), nmr.beginModification()
 *   exportGDCorrectedFID.qs — File + TextStream write pattern
 *   spinSimulationTest.qs   — Application.NMRPredictor, Application.open()
 *   import_process_save_loop.qs — serialization.importFile() batch pattern
 */

/*globals Application, nmr, NMRSpectrum, File, TextStream, Dir, print, serialization*/
/*jslint plusplus: true, indent: 4*/

function spinhanceBatch() {
    "use strict";

    // ── 1. Read config JSON ───────────────────────────────────────────────────
    var configPath = Dir.temp() + "/spinhance_batch_config.json";
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
    dir.setFilter(Dir.Files);
    dir.setNameFilters(["*.xml"]);
    var files = dir.entryList();   // array of filenames (not full paths)

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

    // ── 4. Quit MNova after batch ─────────────────────────────────────────────
    Application.mainWindow.close();
}


/**
 * Open one XML, extract the simulated spectrum, write to outPath.
 * Returns true on success.
 */
function _processOne(xmlPath, outPath) {
    "use strict";

    // Open the spin-system XML — MNova runs the QM simulation automatically
    var doc = Application.open(xmlPath);
    if (!doc) {
        print("  WARN: Application.open() returned null for " + xmlPath);
        return false;
    }

    // Give the simulation thread a moment to finish
    // (Application.waitAllThreadsFinished() if available, else use processEvents)
    if (typeof Application.waitAllThreadsFinished === "function") {
        Application.waitAllThreadsFinished();
    } else {
        Application.processEvents();
    }

    // Get the simulated spectrum
    var rawSpec = nmr.activeSpectrum();
    if (!rawSpec || !rawSpec.isValid()) {
        print("  WARN: no valid activeSpectrum for " + xmlPath);
        doc.close(false);
        return false;
    }

    var spec    = new NMRSpectrum(rawSpec);
    var nPoints = spec.count();

    if (nPoints === 0) {
        print("  WARN: spectrum has 0 points for " + xmlPath);
        doc.close(false);
        return false;
    }

    // ── Write intensity array, one value per line ─────────────────────────────
    var f = new File(outPath);
    if (!f.open(File.WriteOnly | File.Text)) {
        print("  ERROR: cannot write to " + outPath);
        doc.close(false);
        return false;
    }
    var stream = new TextStream(f);
    for (var j = 0; j < nPoints; j++) {
        stream.writeln(spec.real(j));
    }
    f.close();

    doc.close(false);   // close WITHOUT saving a .mnova file
    print("  ok (" + nPoints + " pts → " + outPath + ")");
    return true;
}
