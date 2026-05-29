/**
 * batch_simulate.qs
 * -----------------
 * MestReNova JavaScript batch script.
 * Loads mnova-spinsim XML files, runs the spin simulation, and exports
 * the intensity array as a plain-text column file (.txt) per spectrum.
 *
 * USAGE (headless, from terminal):
 *   /Applications/MestReNova.app/Contents/MacOS/MestReNova \
 *       --no-gui -sf /path/to/batch_simulate.qs \
 *       -- /path/to/xml_dir /path/to/output_dir
 *
 * OUTPUT per XML:
 *   <output_dir>/<stem>.txt   — one intensity value per line (16384 values)
 *   The ppm axis is NOT written here; reconstruct from (from, to, points)
 *   stored in each XML or inferred from the standard 0–12 ppm / 16384 pt grid.
 *
 * NOTES:
 *   • MNova scripting uses Qt's JavaScript engine (QtScript / QJSEngine).
 *   • scriptArgs[0] is the script path itself; user args start at index 1.
 *   • If --no-gui is unsupported on your MNova version, run in GUI mode via
 *     Tools > Scripts > Run Script and pass paths as hard-coded variables below.
 *   • Tested against MNova 14.x–15.x on macOS.
 */

// ── Argument parsing ──────────────────────────────────────────────────────────
var xmlDir    = "";
var outputDir = "";

if (typeof scriptArgs !== "undefined" && scriptArgs.length >= 3) {
    // scriptArgs: [scriptPath, xmlDir, outputDir]
    xmlDir    = scriptArgs[1];
    outputDir = scriptArgs[2];
} else {
    // FALLBACK: hard-code paths here for GUI-mode execution
    xmlDir    = "/Users/labounader/Documents/Claude/Projects/spinhance/data/processed/xmls";
    outputDir = "/Users/labounader/Documents/Claude/Projects/spinhance/data/processed/spectra";
}

// ── Helpers ───────────────────────────────────────────────────────────────────

function ensureDir(path) {
    var dir = new Dir(path);
    if (!dir.exists()) {
        dir.mkpath(path);
    }
}

function listXMLFiles(dirPath) {
    var dir = new Dir(dirPath);
    dir.setFilter(Dir.Files);
    dir.setNameFilters(["*.xml"]);
    return dir.entryList();   // returns array of filenames (not full paths)
}

function writeTextFile(filePath, lines) {
    var f = new File(filePath);
    if (!f.open(File.WriteOnly | File.Text)) {
        print("ERROR: cannot open for writing: " + filePath);
        return false;
    }
    var ts = new TextStream(f);
    for (var i = 0; i < lines.length; i++) {
        ts.writeln(lines[i]);
    }
    f.close();
    return true;
}

// ── Core simulation + export for a single XML ─────────────────────────────────

function simulateAndExport(xmlPath, outPath) {
    // 1. Open the spin simulation XML — MNova auto-runs the simulation on open
    var doc = Application.open(xmlPath);
    if (!doc) {
        print("WARN: failed to open " + xmlPath);
        return false;
    }

    // 2. Wait for simulation to complete (required in headless mode)
    Application.waitAllThreadsFinished();

    // 3. Navigate to the first spectrum item in the document
    var page = doc.currentPage();
    if (!page) {
        print("WARN: no page in document for " + xmlPath);
        doc.close(false);
        return false;
    }

    // Activate the first NMR item on the page
    var item = page.item(0);
    if (!item) {
        print("WARN: no items on page for " + xmlPath);
        doc.close(false);
        return false;
    }
    page.setActiveItem(item);

    var spec = ActiveSpectrum;
    if (!spec || spec.isNull()) {
        print("WARN: ActiveSpectrum is null for " + xmlPath);
        doc.close(false);
        return false;
    }

    // 4. Extract intensity array (real channel)
    var data = spec.realData();   // Float64Array or JS Array of intensities
    if (!data || data.length === 0) {
        print("WARN: empty spectrum data for " + xmlPath);
        doc.close(false);
        return false;
    }

    // 5. Write one value per line
    var lines = [];
    for (var i = 0; i < data.length; i++) {
        lines.push(data[i].toString());
    }
    var ok = writeTextFile(outPath, lines);

    // 6. Close document WITHOUT saving (no .mnova file written)
    doc.close(false);

    return ok;
}

// ── Main batch loop ───────────────────────────────────────────────────────────

function main() {
    ensureDir(outputDir);

    var files = listXMLFiles(xmlDir);
    if (files.length === 0) {
        print("No XML files found in: " + xmlDir);
        return;
    }

    print("Found " + files.length + " XML files. Starting batch simulation...");

    var succeeded = 0;
    var failed    = 0;

    for (var i = 0; i < files.length; i++) {
        var fname   = files[i];
        var xmlPath = xmlDir + "/" + fname;

        // Derive output filename: strip .xml, add .txt
        var stem    = fname.replace(/\.xml$/i, "");
        var outPath = outputDir + "/" + stem + ".txt";

        print("[" + (i + 1) + "/" + files.length + "] " + fname + " ...");

        var ok = simulateAndExport(xmlPath, outPath);
        if (ok) {
            succeeded++;
        } else {
            failed++;
            print("  FAILED: " + fname);
        }
    }

    print("");
    print("Done. Succeeded: " + succeeded + "  Failed: " + failed);
}

main();
