import sys
from flask import Flask, request, send_file, jsonify,render_template
from flask_cors import CORS
import os
from werkzeug.utils import secure_filename
import subprocess
import numpy as np
import pickle

app = Flask(__name__)
CORS(app)

frameindex=0
@app.route('/home', methods=['GET','POST'])
def wel():
    return render_template('0.html')
@app.route('/1.html', methods=['GET','POST'])
def homet():
    return render_template('1.html')

def process_video(video_path, text_message):
    # Process the video path and text message as needed
    print("Received video path:", video_path)
    print("Received text message:", text_message)

    # Perform any processing or manipulation of the video or text message here
    # For demonstration purposes, let's just return the video path
    return video_path

def caesar_cipher_encrypt(message, key):
    encrypted_message = ""
    for char in message:
        if char.isalpha():
            shift = ord('A' if char.isupper() else 'a')
            encrypted_char = chr((ord(char) - shift + key) % 26 + shift)
            encrypted_message += encrypted_char
        else:
            encrypted_message += char
    return encrypted_message

def caesar_cipher_decrypt(ciphertext, key):
    decrypted_message = ""
    for char in ciphertext:
        if char.isalpha():
            shift = ord('A' if char.isupper() else 'a')
            decrypted_char = chr((ord(char) - shift - key) % 26 + shift)
            decrypted_message += decrypted_char
        else:
            decrypted_message += char
    return decrypted_message

# Reference DNA numerically
reference_DNA = {'00': 'A', '01': 'T', '10': 'C', '11': 'G'}

# Function to convert binary to fake DNA
def binary_to_fake_DNA(binary_string):
    fake_DNA = ''
    for i in range(0, len(binary_string), 2):
        pair = binary_string[i:i+2]
        fake_DNA += reference_DNA[pair]
    return fake_DNA

# Function to apply DNA complementary rule to binary string
def apply_DNA_complementary_rule(binary_string):
    complement_pairs = {'0': '1', '1': '0'}
    complemented_binary_string = ''.join(complement_pairs[bit] for bit in binary_string)
    return complemented_binary_string

# Function to encrypt cipher text to fake DNA
def encrypt_to_fake_DNA(cipher_text):
    binary_text = ''.join(format(ord(char), '08b') for char in cipher_text)
    complemented_binary_text = apply_DNA_complementary_rule(binary_text)
    fake_DNA_text = binary_to_fake_DNA(complemented_binary_text)
    return fake_DNA_text
# Convert DNA sequence to binary string
def dna_to_binary(dna_sequence):
    binary_string = ""
    for nucleotide in dna_sequence:
        if nucleotide == 'A':
            binary_string += '00'
        elif nucleotide == 'T':
            binary_string += '01'
        elif nucleotide == 'C':
            binary_string += '10'
        elif nucleotide == 'G':
            binary_string += '11'
    return binary_string



##############dcrp###########
# Function to convert fake DNA to binary
def fake_DNA_to_binary(fake_DNA):
    binary_string = ''
    for char in fake_DNA:
        for key, value in reference_DNA.items():
            if value == char:
                binary_string += key
                break
    return binary_string

# Function to apply DNA complementary rule to binary string
def apply_inverse_DNA_complementary_rule(binary_string):
    complemented_pairs = {'0': '1', '1': '0'}
    complemented_binary_string = ''.join(complemented_pairs[bit] for bit in binary_string)
    return complemented_binary_string

# Function to decrypt fake DNA to cipher text
def decrypt_fake_DNA(encrypted_fake_DNA):
    binary_text = fake_DNA_to_binary(encrypted_fake_DNA)
    complemented_binary_text = apply_inverse_DNA_complementary_rule(binary_text)
    decrypted_text = ''
    for i in range(0, len(complemented_binary_text), 8):
        byte = complemented_binary_text[i:i+8]
        decrypted_text += chr(int(byte, 2))
    return decrypted_text
def frame_selection(video_path):
    # Open the video
    cap = cv2.VideoCapture(video_path)

    # Check if video opened successfully
    if not cap.isOpened():
        print("Error: Unable to open video")
        return None

    # Initialize list to store indices of complex frames
    complex_frame_indices = []

    # Read the first frame
    ret, prev_frame = cap.read()

    # Convert the first frame to grayscale
    prev_frame_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    frame_index = 0

    # Loop through the video frames
    while True:
        # Read the next frame
        ret, frame = cap.read()
        if not ret:
            break

        # Convert the frame to grayscale
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Calculate Discrete Cosine Transform (DCT) of the frames
        dct_prev = cv2.dct(np.float32(prev_frame_gray))
        dct_curr = cv2.dct(np.float32(frame_gray))

        # Calculate mean of the DCT coefficients
        mean_prev = np.mean(dct_prev)
        mean_curr = np.mean(dct_curr)

        # Check if the means of the DCT coefficients of consecutive frames are different
        if abs(mean_curr - mean_prev) > 0.001:
            complex_frame_indices.append(frame_index)

        # Update previous frame and frame index
        prev_frame_gray = frame_gray.copy()
        frame_index += 1

    # Release the video capture object
    cap.release()

    return complex_frame_indices


def embed_data_in_frame(frame, data, pixel_coordinates_file, frame_index):
    # Define a mapping from data bits to integer values
    data_mapping = {0: 0, 1: 1}
    cv2.imwrite('static/selframe.jpg', frame) 

    # Flatten the frame into a 1D array
    flattened_frame = frame.flatten()

    # Save the selected pixel coordinates to a file
    #with open(pixel_coordinates_file, 'a') as f:
        #f.write(f"Frame {frame_index}:\n")

    data_index = 0
    for pixel_value in flattened_frame:
        # Break if all data bits have been embedded
        if data_index >= len(data):
            break

        # Extract the least significant bit of the pixel value
        lsb = pixel_value & 1

        # Get the next data bit to embed
        data_bit = data_mapping.get(int(data[data_index]))
        if data_bit is not None:
            # Embed the data bit into the least significant bit of the pixel value

            new_pixel_value = (pixel_value & ~1) | data_bit
            print(pixel_value,new_pixel_value)
            flattened_frame[data_index] = new_pixel_value

            # Write the pixel coordinates to the file
            with open(pixel_coordinates_file, 'a') as f:
                f.write(f"{data_index % frame.shape[1]},{data_index // frame.shape[1]}\n")

            data_index += 1
        else:
            print("Invalid data encountered:", data[data_index])

    # Reshape the flattened frame back to its original shape
    modified_frame = flattened_frame.reshape(frame.shape)

    return modified_frame

import cv2
data_length=0
def embedding(video_path,message,key):

# Open the input video file
#video_path='/content/drive/MyDrive/big_buck_bunny_720p_1mb.mp4'
    indices=frame_selection(video_path)

    input_video = cv2.VideoCapture(video_path)

    # Get the video properties
    fps = input_video.get(cv2.CAP_PROP_FPS)
    frame_width = int(input_video.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(input_video.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Define the codec and create VideoWriter object
    

   

    fourcc = cv2.VideoWriter_fourcc('F','F','V','1')  # Codec for MP4 files
    output_video = cv2.VideoWriter('static/output.avi', fourcc, fps, (frame_width, frame_height))

    fourcc1 = cv2.VideoWriter_fourcc('M','P','4','')  # Codec for MP4 files
    output_video1 = cv2.VideoWriter('static/output.mp4', fourcc1, fps, (frame_width, frame_height))

    # Read and write frames
    frame_index=indices[0]

    print("frame index: ", frame_index)
    i=0
    #binary_sequence='10101011'
    key =int(key)
    c_text = caesar_cipher_encrypt(message, key)
    print('ct',c_text)
    DNA_sequence = encrypt_to_fake_DNA(c_text)
    binary_sequence = dna_to_binary(DNA_sequence)
    data_length=len(binary_sequence)
    print(data_length)
    
    with open('myfile', 'w') as f:
        f.write(str(frame_index)+'\n')
        f.write(str(data_length)+'\n')
      

    while input_video.isOpened():
        ret, frame = input_video.read()  # Read a frame
        if not ret:
            break  # Break the loop if there are no more frames

        if i==indices[0]:
            frame_with_dna = embed_data_in_frame(frame.copy(), binary_sequence, 'px.txt', frame_index)
            output_video.write(frame_with_dna)
            output_video1.write(frame_with_dna)

        # Process the frame if needed
        # For example, you can apply some operations on the frame here
        else:
            output_video.write(frame)  # Write the frame to the output video
            output_video1.write(frame)
        i+=1
    print("done")
    # Release resources
    print(binary_sequence)
    input_video.release()
    output_video.release()
    output_video1.release()
    cv2.destroyAllWindows()
    return c_text
def extract_data_from_frame_ver(frame,data_length):
    data_mapping = {0: 0, 1: 1}
    #data_length=40

    # Flatten the frame into a 1D array
    flattened_frame = frame.flatten()
    ebin=''
    cnt=0
    for pixel_value in flattened_frame:
      #print(pixel_value)
      lsb=pixel_value & 1
      ebin+=str(lsb)
      cnt+=1
      if cnt==data_length:
        return ebin
import cv2

def extraction(video_path,frameindex,data_length,key):

# Open the input video file
    input_video = cv2.VideoCapture(video_path)

    # Get the video properties
    fps = input_video.get(cv2.CAP_PROP_FPS)
    frame_width = int(input_video.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(input_video.get(cv2.CAP_PROP_FRAME_HEIGHT))



    # Read and write frames
    frame_index=2
    i=0
    #binary_sequence='10101011'
    #data_length=len(binary_sequence)
    while input_video.isOpened():
        ret, frame = input_video.read()  # Read a frame
        if not ret:
            break  # Break the loop if there are no more frames

        if i==frameindex:
            eseq= extract_data_from_frame_ver(frame,data_length)
            print(eseq)

        # Process the frame if needed
        # For example, you can apply some operations on the frame here
        else:
            pass  # Write the frame to the output video
        i+=1

    # Release resources
    input_video.release()

    cv2.destroyAllWindows()
    complemented_binary_text = apply_inverse_DNA_complementary_rule(eseq)
    ciphertext = ''
    for i in range(0, len(complemented_binary_text), 8):
        byte = complemented_binary_text[i:i+8]
        ciphertext += chr(int(byte, 2))
    print(ciphertext)
    #key=32
    key=int(key)
    original_message = caesar_cipher_decrypt(ciphertext, key)
    print("Decrypted message:", original_message)
    return original_message

@app.route('/encrypt', methods=['POST'])
def encrypt():
        f = request.files['ifile']   
        key=request.form.get('key')    
        filename = secure_filename(f.filename)
        rimg=filename
        f.save("static/" + rimg)
        message=request.form.get('message')
        cipher=embedding('static/'+rimg,message,key)
        file11 = open('myfile', 'r')
        Lines = file11.readlines()
        frameindex=Lines[0]
        return '''
       <p style="font-size: 1.2em; color: #333;">
        Selected video frame with index '''+frameindex+'''
    </p>
    <img src="static/selframe.jpg" width="320" height="240"  padding: 5px; margin-bottom: 20px;">
    
    <p style="font-size: 1.2em; color: #333;">
        Stego video
    </p>
    <video width="320" height="240" controls  margin-bottom: 20px;">
        <source src="static/output.mp4" type="video/mp4">
        Your browser does not support the video tag.
    </video>
    
    <p style="font-size: 1.2em; color: #333;">
        Encrypted sequence: '''+cipher      
@app.route('/decrypt', methods=['GET','POST'])
def decrypt():
    key=request.form.get('key')  
    file1 = open('myfile', 'r')
    Lines = file1.readlines()
    frameindex=int(Lines[0])
    data_length=int(Lines[1])
    o=extraction('static/output.avi',frameindex,data_length,key)
    return o

if __name__ == '__main__':
    app.run(port=5500, debug=True)
