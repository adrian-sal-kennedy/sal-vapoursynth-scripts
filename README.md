# sal-vapoursynth-scripts
A collection of my vapoursynth tools and some shell stuff to automate their use.

saltools.py contains functions for processing video including:
- noise reduction
- framerate interpolation
- motion-blur
- motion stabilization
- field matching based on simple adjacent field comprison
- cleanup of any residual combing (such as with analog video with timebase errors or video with slow fades below combing thresholds)
- deinterlace
- removal of phasing caused by LED lighting on rolling shutter (must specify a period by counting wave lengths visually)

framestats.py contains tools for video analysis and measurement allowing automated quality control and logging.
