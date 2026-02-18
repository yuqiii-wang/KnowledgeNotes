# ComfyUI Audio & Video Lip Sync Workflow

This guide details high-quality workflows for generating lip-synced videos from audio and images using ComfyUI.

## Prerequisites

To follow these workflows, you will need **ComfyUI Manager** to install the following custom node packs:

1.  **ComfyUI-VideoHelperSuite (VHS)**: Essential for loading video and combining video + audio.
2.  **ComfyUI-EchoMimic** (or **ComfyUI-Hallo**): For state-of-the-art audio-driven animation.
3.  **ComfyUI-LivePortraitKJ**: For high-quality face retargeting and expression control.
4.  **ComfyUI-Wav2Lip** (Optional but reliable): For generating "driver" videos if EchoMimic is too heavy.

---

## Workflow 1: High-Fidelity Audio-Driven Animation (EchoMimic)

**Best for:** Creating a talking head video *from scratch* using just a single static image and an audio file.

EchoMimic is currently one of the top performers for generating lifelike motion and lip-sync from audio.

### Step-by-Step

1.  **Load Source Image**
    *   **Node:** `Load Image`
    *   **Input:** Your target face (square aspect ratio 1:1, usually 512x512 or 768x768 is best).

2.  **Load Audio**
    *   **Node:** `Load Audio` (standard) or `VHS_LoadAudio`
    *   **Input:** Your speech file (.wav/.mp3). Note the **duration** (e.g., 10 seconds).

3.  **Sampler Configuration (EchoMimic)**
    *   **Node:** `EchoMimicSampler`
    *   **Inputs:**
        *   `ref_image`: Connect to Step 1.
        *   `driving_audio`: Connect to Step 2.
    *   **Settings:**
        *   `width`: 512 / 768
        *   `height`: 512 / 768
        *   `length`: Set frame count = `audio_duration_seconds * fps`.
            *   *Tip:* Standard is 24fps or 25fps. For 10s audio, set `length` to ~240.
        *   `steps`: 25-30
        *   `cfg`: 2.5 - 3.5 (Higher values can cause jitter).
        *   `context_frames`: 12-24 (Helps with consistency).

4.  **VAE Decode**
    *   **Node:** `VAE Decode`
    *   **Input:** `samples` from EchoMimic.
    *   **Output:** `images` (pixel frames).

5.  **Face Restoration (Optional but Recommended)**
    *   **Node:** `FaceDetailer` (from Impact Pack)
    *   **Input:** `image` (from Decode).
    *   **Model:** `bbox/face_yolov8m`.
    *   **Settings:** `denoise` 0.25 - 0.35.
    *   *Purpose:* Sharpens eyes and teeth that might get blurry during generation.

6.  **Combine & Save (Sync Critical Step)**
    *   **Node:** `VHS_VideoCombine`
    *   **Inputs:**
        *   `images`: From FaceDetailer (or Decoder).
        *   `audio`: **Crucially**, connect the *original* audio from Step 2 here.
    *   **Settings:**
        *   `frame_rate`: Match the FPS used in Step 3 (e.g., 24).
        *   `format`: `video/h264-mp4`.
    *   *Result:* A perfectly synced MP4 with audio.

---

## Workflow 2: The "Hybrid" Refinement (LivePortrait)

**Best for:** Driving a high-quality (or stylized) image with precise expression control, or fixing a "mushy" mouth from another video.

This method uses a cheaper/older model (like Wav2Lip) to generate the *motion*, and then uses LivePortrait (which is excellent at preserving identity and quality) to render the final pixels.

### Step-by-Step

1.  **Generate "Driver" Video**
    *   Use **Wav2Lip** or **SadTalker** to create a low-quality video where the lips move correctly to your audio.
    *   *Note:* The face doesn't need to look good here, only the *lips* need to move well.

2.  **Load High-Quality Target**
    *   **Node:** `Load Image`
    *   **Input:** The high-res/stylized face you actually want to show.

3.  **Setup LivePortrait**
    *   **Node:** `LivePortraitLoadCropper` -> `LivePortraitProcess` -> `LivePortraitComposite`.
    *   **Inputs:**
        *   `source_image`: Your high-quality target (Step 2).
        *   `driving_video`: Your rough lip-sync video (Step 1).
    *   **Settings (LivePortraitProcess):**
        *   `lip_retargeting`: **Enable**. This forces the target's lips to mimic the driver's lips exactly.
        *   `lip_retargeting_multiplier`: `1.0` (Default).
            *   Increase to `1.2` if the speaker is mumbling.
            *   Decrease to `0.8` if the mouth opens too wide.
        *   `relative_motion`: `False` (Usually stable for talking heads) or `True` (if you want head movement from the driver video).

4.  **Output**
    *   **Node:** `VHS_VideoCombine`
    *   **Input:** Result from `LivePortraitComposite`.
    *   **Audio:** Connect the original audio track.

---

## Synchronization Best Practices

### 1. Match Frame Rates
Drift happens when the audio length and video frame count don't align.
*   **Formula:** `Total Frames = Audio Seconds * FPS`
*   **Standard:** Use **24 fps** or **25 fps** for most realistic talk.
*   **Avoid:** 30fps or 60fps unless your model specifically supports it (most diffusion models train on lower fps).

### 2. Audio Sample Rate
Ensure your input audio is standard (44.1kHz or 48kHz). Strange sample rates (like 22kHz) can sometimes confuse the FFmpeg backend in `VHS_VideoCombine`.

### 3. "Speak Sync" (Alignment)
If the mouth moves *before* the sound:
*   In `VHS_VideoCombine`, there isn't a simple offset.
*   **Fix:** Add `0.1s` of silence to the start of your audio file *before* loading it into ComfyUI, OR trim the first few video frames using a `ImageSelector` node.

### 4. Expression Control
*   **Blinking:** If your generated video feels "dead" (staring), use **LivePortrait's** `auto_blink` toggle (if available in your specific node version) or drive it with a video of a person blinking naturally.
*   **Teeth/Tongue:** If these look weird, use `FaceDetailer` with a `mesh` precision (if available) or simply keep `denoise` low (`<0.3`) so it doesn't hallucinate extra teeth.
