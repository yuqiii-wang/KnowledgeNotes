# MP4

MP4 is a container format, not a video or audio format itself.

The "Documents" (Codecs): The actual video and audio inside the MP4 container are compressed using a codec (short for coder-decoder).

* Video Codec: Usually H.264 (AVC) or H.265 (HEVC).
* Audio Codec: Usually AAC (Advanced Audio Coding).

An MP4 file is built from a series of objects called "atoms", of which two typical are

* `moov` (Movie Atom): contains all the metadata needed to understand and play the media.
* `mdat` (Media Data Atom): containS all the actual, raw compressed video and audio samples.