const mjAPI = require("mathjax-node");
const argparse = require("argparse");
const csv = require("csv");
const fs = require("fs");
const sharp = require("sharp");
const path = require("path");

let parser = argparse.ArgumentParser();
parser.addArgument("csv");
parser.addArgument("outdir");
parser.addArgument("--size", {nargs:2, defaultValue:[300, 300], type:'int'});
parser.addArgument("--pad", {defaultValue:5, type:'int'});
let args = parser.parseArgs();

fs.existsSync(args.outdir) || fs.mkdirSync(args.outdir);

fs.createReadStream(args.csv)
    .pipe(csv.parse())
    .pipe(csv.transform(data => {
        let [id, latex] = data;
        return [id, {
            math: latex,
            format: 'TeX',
            svg: true,
        }];
    }))
    .pipe(csv.transform(config => {
        let [id, options] = config;
        let outPath = path.join(args.outdir, id + '.png');
        if (!fs.existsSync(outPath)) {
            mjAPI.typeset(options, result => {
                if (result.errors) {
                    console.log("Render " + id + " error: " + options.math);
                } else {
                    let buffer = Buffer.from(result.svg);
                    sharp(buffer, { density: 1000 })
                        .resize({
                            width: args.size[0],
                            height: args.size[1],
                            fit: 'contain',
                            background: 'white',
                        })
                        .extend({
                            top: args.pad,
                            bottom: args.pad,
                            left: args.pad,
                            right: args.pad,
                            background: 'white'
                        })
                        .flatten({ background: 'white' }) // merge alpha
                        .toFile(outPath);
                }
            });
        }
    }));
