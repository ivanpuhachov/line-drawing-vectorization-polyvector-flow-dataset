#target "estoolkit"

//authors: Dima Smirnov, MIT, 2018, Mikhail Bessmeltsev, UdeM, 2020, Ivan Puhachov, UdeM, 2021

var basePath = 'C:\\FOLDER_QUICK\\example\\';  // absolute path to the current folder
var svgPath = basePath + 'svg\\'; // path for svg files
var templatePath = basePath + 'template_mybrush.ai'; // file with adobe illustrator TEMPLATE
var processedPath = basePath + 'processed_svg\\'; // path to move svg file once it is processed
var outpngPath = basePath + 'png\\'; // path to output rendered png files
var defaultpngPath =  basePath + 'default_png\\'; // path to put rendered png with standard brush
var outwidthsPath = basePath + 'widths\\'; // path to output stroke widths

function processAi() {
    // check that Adobe Illustrator can open SVG files
    var folder = new Folder(svgPath); 
    var files = folder.getFiles("*.svg");
    if (files.length > 0) {
        var bt = new BridgeTalk();
        bt.target = 'illustrator';
        bt.body = process.toSource() + '();';
        bt.onResult = function (resultMsg) {
            while (BridgeTalk.getStatus('illustrator') != 'ISNOTRUNNING') $.sleep(20);
            processAi();
        }
        bt.send();
    }
}

function approxGaussian() {
    // returns approximately gaussian between -1 and 1
    // used to create random stroke width 
    var rand = 0;
    var n_experiments = 3;
    for (var i=0; i<n_experiments; i+=1) {
        rand += Math.random();
        }
    return (2*rand - n_experiments) / n_experiments;
    }

function openFile(fileObj, encoding) {
    // reads svg file
    encoding = encoding || "utf-8";
    fileObj = (fileObj instanceof File) ? fileObj : new File(fileObj);
    var parentFolder = fileObj.parent;
    if (!parentFolder.exists && !parentFolder.create())
    
       throw new Error("Cannot create file in path " + fileObj.fsName);
    fileObj.encoding = encoding;
    fileObj.open("w");
    return fileObj;
}

function process() {

    var folder = new Folder(svgPath);
    var files = folder.getFiles("*.svg"); // query all SVG file from svgPath directory
    var nProcessed = 0;
    var doc = app.open(new File(templatePath)); // open template
    var strokeColor = new RGBColor();

    var exportOps = new ExportOptionsPNG24();
    exportOps.artBoardClipping = true;
    exportOps.transparency = false;
    // exportOps.horizontalScale = exportOps.verticalScale = 128/625 * 100;

    for (var i = 0; nProcessed < 500 && i < files.length; i++) // loop for at most 500 files (warning: bigger loop leads to memory leak and painfully slow execution! don't do more than 500)
    {
        var inputFile = new File(files[i]);
        //$.writeln(files[i]);
        var filename = inputFile.name.substring(0, inputFile.name.lastIndexOf('.')); // extract filename
        
        var svg = doc.groupItems.createFromFile(inputFile); // place svg strokes on template canvas

        svg.position = [(288-svg.width)/2, -(288-svg.height)/2]; // place svg file in the center of 288x288 image
        //svg.position = [0, 0] does not work as expected

        for (j=0; j < doc.brushes.length; j++){ // loop over all brushes in template file
        //for (j=0; j < 1; j++){ //debug loop, comment the previous line
            var pngFilename = filename + '_' + j + '.png';
            var JFile = new File(outwidthsPath+pngFilename + '.txt');
            openFile(JFile);            
            
            //var b = Math.floor(Math.random() * doc.brushes.length);
            var brush = doc.brushes[j];
            //$.writeln(brush);
            strokeColor.red = strokeColor.green = strokeColor.blue = 0; // Math.floor(Math.random() * 50);
                for (l=0; l < svg.pathItems.length; l++) { // for each stroke in svg file
                    var strokeWidth = 0.35 + approxGaussian()*0.1; // set the stroke width
                    JFile.write(strokeWidth + '  '); // write it to file
                    brush.applyTo(svg.pathItems[l]); // apply brush and render
                    svg.pathItems[l].strokeWidth = strokeWidth;
                    svg.pathItems[l].strokeColor = strokeColor;
                }
             
             // render PNG 
            doc.exportFile(new File(outpngPath + pngFilename), ExportType.PNG24, exportOps);
            
            if (j==0){ // brush 0 is a default uniform brush, store it separately
                 pngFilename = filename + '.png';
                 for (l=0; l < svg.pathItems.length; l++) {
                    var strokeWidth = 0.15;
                    JFile.write(strokeWidth + '  ');
                    brush.applyTo(svg.pathItems[l]);
                    svg.pathItems[l].strokeWidth = strokeWidth;
                    svg.pathItems[l].strokeColor = strokeColor;
                }
                 doc.exportFile(new File(defaultpngPath + pngFilename), ExportType.PNG24, exportOps);
                }
            JFile.close();
        }

        svg.remove(); // clear template file from strokes
        
        inputFile.copy(processedPath + inputFile.name); // copy SVG file to processedPath folder
        inputFile.remove(); // remove original SVG file
        
        nProcessed++;
        $.writeln(nProcessed) // output file ID to track progress
    }
    doc.close(SaveOptions.DONOTSAVECHANGES);
    app.quit(); // close adobe illustrator (to free memory)
}

function checkFolder(x) {
    // check that 
    var  MyFolder = new Folder(x);
    $.writeln(x);
     if (!(MyFolder.exists)) {
         $.writeln("--> FOLDER NOT FOUND! ", x, " <--");
         return false
     } else {
         return true
     }
}

function main() {
    // check files and folder exists
    var MyFile = new File (templatePath);
    if  (!(MyFile.exists)) {
        $.writeln(templatePath);
        $.writeln("-> TEMPLATE FILE NOT FOUND!");
        return "FAILED"
        }
    if (!(checkFolder(svgPath))) {
        return "FAILED"
        }
    if (!(checkFolder(outpngPath))) {
        return "FAILED"
        }
    if (!(checkFolder(outwidthsPath))) {
        return "FAILED"
        }
    if (!(checkFolder(processedPath))) {
        return "FAILED"
        }
    if (!(checkFolder(defaultpngPath))) {
        return "FAILED"
        }

    // do the job
    processAi(); // check that SVG folder is non-empty 
    process();
    return "DONE"
    }

main();
