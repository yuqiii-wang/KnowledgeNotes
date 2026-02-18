# Comfy UI Audio


## Workflow Comparison Summary

| Feature | **Audio-First (Reactive)** | **Video-First (Dubbing)** | **Parallel (Combined)** | **Audio Only (Composition)** |
| :--- | :--- | :--- | :--- | :--- |
| **Concept** | *Video follows Audio* | *Audio follows Video* | *Convergent Synthesis* | *Pure Audio Synthesis* |
| **Primary Input** | Audio File (Waveform) | Video Latents / Images | Text Prompt (Dual) | Text Prompt (Lyrics) |
| **Mechanism** | Amplitude/FFT analysis modulates video params (CFG, Motion). | Visual content captioning (VQA) prompts audio generation. | Independent generation chains synced by frame/sample rate. | LLM Lyrics $\to$ TTS Vocals + Text-to-Music $\to$ Mixer. |
| **Speak Sync** | **Low** (Not focused on lip-sync). | **Medium** (Loose timing for dialogue). | **Low** (Thematic match only). | **Internal** (Vocals match backing track BPM). |
| **Background Sound Sync** | **High** (Frame-perfect beat matching). | **Medium** (Semantic match for Foley). | **Low** (Thematic match only). | **N/A** (Audio only). |
| **Use Case** | Music Videos, Visualizers. | Sound Effects (Foley), Dialogue. | Mood pieces, Backgrounds. | Songs, Podcasts, Radio. |

