# Keypoint-Driven Line Drawing Vectorization via PolyVector Flow -- dataset generation

![Example](/png/test_50_8.png)

> TLDR: dataset was generated from vector drawings ("[Quick, Draw](https://github.com/googlecreativelab/quickdraw-dataset)" and "[Creative Birds](https://songweige.github.io/projects/creative_sketech_generation/home.html)"

Subset: (500 MB, 40 000 images) [google drive link](https://drive.google.com/file/d/1KQrLGW82Fk_bYuB8uC0DzSWafDj97gLn/view?usp=sharing)
Full dataset: (14 GB, 927 000 images) [onedrive link](https://udemontreal-my.sharepoint.com/:u:/g/personal/ivan_puhachov_umontreal_ca/EVGygX7l-bBCjwANyPgcIB4BNdzMytMBR8-JcR4J4GyzQw?e=OhA2QD)

## Processing ndjson
See `generate_quickdraw.py` and `generate_quickdraw_svg.py`. Given path to ndjson, it dumps corresponding `svg` file and `npz` keypoints data to the output folder.

### Requirements
**Python**: ndjson, svgpathtools, shapely, [simplification](https://pypi.org/project/simplification/)

## Data Generation

> **TLDR**: Adobe Illustrator + Adobe ExtendScript   
**Requirements**: Windows, Adobe Illustrator

 * Download ExtendScript from github https://github.com/Adobe-CEP/CEP-Resources/tree/master/ExtendScript-Toolkit
 * Javascript documentation for adobe illustrator https://ai-scripting.docsforadobe.dev/

### Bugs and Warnings
 * 32-bit only
 * Data leak detected, rendering speed decreases. We recommend reload the program after 500 images (hence loop limits in the code)

 * Bug: your template file (`template_mybrush.ai`) should be cleared
![Example](/instructions/bug1.png)

### Instructions
1. Open ExtendScript. **Update** `basePath` variable
![Example](/instructions/1.png)
2. Link with Illustrator (it will load the program)
![Example](/instructions/2.png)
3. Once you run the script, Illustrator may freeze
![Example](/instructions/3.png)
4. Observe the progress in Console (top right)
![Example](/instructions/4.png)


##### Citation
```
@article{Puhachov2021KeypointPolyvector,
    author = {Ivan Puhachov and William Neveu and Edward Chien and Mikhail Bessmeltsev},
    title = {Keypoint-Driven Line Drawing Vectorization via PolyVector Flow},
    journal = {ACM Transactions on Graphics (Proceedings of SIGGRAPH Asia)},
    volume = {40}, number = {6}, year = {2021}, month = dec,
    doi = {10.1145/3478513.3480529}
}
```
