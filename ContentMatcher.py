import cv2
from moviepy.editor import VideoFileClip
import numpy as np
import librosa
from scipy.signal import correlate
from scipy.spatial.distance import cosine

print("Pleas wait for all the task to be done ")
###########Audio detacher #######################
# Load the Video
video_main = VideoFileClip("main.mp4")
video_cut_part = VideoFileClip("cut-part.mp4")

# Extract the Audio
audio_main = video_main.audio
audio_cutpart = video_cut_part.audio
#
# #Export the Audio
audio_main.write_audiofile("main.wav")
audio_cutpart.write_audiofile("cut_part.wav")

#########video sound remover ###############

# video_main_sound_less = VideoFileClip("main.mp4")
# video_cut_part_sound_less = VideoFileClip("cut-part.mp4")
# soundless_main_video = video_main_sound_less.without_audio()
# soundless_cut_part_video = video_cut_part_sound_less.without_audio()

# soundless_main_video.write_videofile("final_main_video.mp4")
# soundless_cut_part_video.write_videofile("final_cut-part_video.mp4")


print("Processing audio Similarity!")


###################Audio Matcher#####################


# Load and preprocess audio data
def preprocess_audio(audio_path, num_frames=100):
    y, sr = librosa.load(audio_path, sr=None)
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
    mel_spec = mel_spec.T[:num_frames, :]  # Limit to a fixed number of frames
    return y, mel_spec


# Calculate waveform similarity using cross-correlation
# def calculate_waveform_similarity(waveform1, waveform2):
#     corr = correlate(waveform1, waveform2)
#     max_corr = np.max(corr)
#     return max_corr

def calculate_waveform_similarity(waveform_main, waveform_cutpart):
    corr = correlate(waveform_main, waveform_cutpart)
    max_corr = np.max(corr)
    # print("Raw max_corr:", max_corr)
    normalized_similarity = (max_corr + 1) / 2  # Normalize to the range [0, 1]
    return normalized_similarity


# Calculate spectral similarity using cosine similarity
def calculate_spectral_similarity(spectral1, spectral2):
    similarity = 1 - cosine(spectral1.flatten(), spectral2.flatten())
    return similarity


# Calculate similarity as a percentage
def calculate_similarity_percentage(similarity_value, threshold):
    percentage_similarity = (similarity_value / threshold) * 100
    return percentage_similarity


# Paths to the two audio files
audio_1 = "main.wav"
audio_2 = "cut_part.wav"

# Load and preprocess audio clips
waveform1, mel_spec1 = preprocess_audio(audio_1)
waveform2, mel_spec2 = preprocess_audio(audio_2)

# Calculate waveform similarity
waveform_similarity = calculate_waveform_similarity(waveform1, waveform2)

# Calculate spectral similarity
spectral_similarity = calculate_spectral_similarity(mel_spec1, mel_spec2)

# Set similarity thresholds
waveform_similarity_threshold = 0.7
spectral_similarity_threshold = 0.8

# Calculate similarity as a percentage
# # Compare similarities and determine if the audio files are similar
if spectral_similarity > spectral_similarity_threshold:
    # print("")
    print("The audio files are {0:.2f}% similar.".format(spectral_similarity * 100))
else:
    print("The audio files are not similar.")

print("wait for video's result")

#############Video Matcher###############


# Load main video and template video
main_video = cv2.VideoCapture("main.mp4")
template_video = cv2.VideoCapture("cut-part.mp4")

# Read the first frame of the template video
template_frame = template_video.read()[1]

# Get dimensions of the template video
template_width = template_frame.shape[1]
template_height = template_frame.shape[0]

# Initialize sliding window step size
step_size = 1  # Adjust this as needed

# Initialize counters
total_frames = 0
matched_frames = 0

# Iterate over frames in the main video
while True:
    ret, main_frame = main_video.read()
    if not ret:
        break

    total_frames += 1

    # Resize main frame to template dimensions
    main_frame_resized = cv2.resize(main_frame, (template_width, template_height))

    # Compare template video with current main frame
    similarity_map = cv2.matchTemplate(main_frame_resized, template_frame, cv2.TM_CCOEFF_NORMED)

    # Find maximum similarity location
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(similarity_map)

    # Check if similarity score is above a threshold
    similarity_threshold = 0.8  # Adjust this as needed
    if max_val >= similarity_threshold:
        matched_frames += 1
        # Draw rectangle around matched region
        top_left = max_loc
        bottom_right = (top_left[0] + template_width, top_left[1] + template_height)
        cv2.rectangle(main_frame, top_left, bottom_right, (0, 255, 0), 2)

    # Display frame with template matches
    # cv2.imshow("Template Matching", main_frame)
    if cv2.waitKey(1) == 27:  # Press Esc to exit
        break

main_video.release()
template_video.release()
cv2.destroyAllWindows()

# Calculate and display accuracy
accuracy = ((matched_frames / total_frames) * 100)*100
print(f"number of {accuracy:.2f}% video matched.")

print("all the tasks are done")
print("Thanks for the wait")
