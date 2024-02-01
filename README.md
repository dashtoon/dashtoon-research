# Mis-Alignment Penalization Scheme Between Storyboard and Generated Image Pair

**Problem space**<br>
<br>
Goal is to filter out data points where the generated image is not aligned with the storyboard image. Since generated image was generated with storyboard image as the conditioning input, the objects in generated image need to respect the positions, orientations and shapes of objects in the storyboard image.
Therefore if the generated image fails to do so, we need to filter that pair out.
<br>
This repo contains the code and details of the algorithm and execution flow to achieve above.
<hr>

``run.py`` is the main script which contains code to:
- Extract bounding boxes using [GroundingDINO]([url](https://github.com/IDEA-Research/GroundingDINO)https://github.com/IDEA-Research/GroundingDINO).
- Extract Segmentation Masks with promptable [Segment-Anything]([url](https://github.com/facebookresearch/segment-anything)https://github.com/facebookresearch/segment-anything) using the bounding box outputs from GroundingDINO.
- Calculate object mapping information between storyboard and generated image pair based on segmentation masks and boudning box overlaps.
<br>
Note: The first two steps are encapsulated using an open-source implementation [lang_segment_anything]([url](https://github.com/luca-medeiros/lang-segment-anything)https://github.com/luca-medeiros/lang-segment-anything)
