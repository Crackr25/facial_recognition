import os
import cv2
import subprocess
import numpy as np
from flask import Flask, request, jsonify

app = Flask(__name__)


image_dir = os.path.join(os.path.dirname(__file__), 'images')
classifier = 'haarcascade_frontalface_default.xml'

@app.route('/')
def hello_world():
	return 'Hello, Clyde!'

@app.route('/createfolder', methods=['POST'])
def create_folder():
    # Get the folder name from the request data
    folder_name = request.form.get('folder_name')

    
    if folder_name:
        # Create a folder with the provided name
        folder_path = os.path.join(image_dir, folder_name)
        try:
            os.mkdir(folder_path)
            return jsonify({'message': f'Folder "{folder_name}" created successfully'}), 201
        except FileExistsError:
            return jsonify({'error': f'Folder "{folder_name}" already exists'}), 400
        except Exception as e:
            return jsonify({'error': f'Failed to create folder: {str(e)}'}), 500
    else:
        return jsonify({'error': 'Folder name not provided in the request'}), 400

@app.route('/uploadimages', methods=['POST'])
def upload_images():
    # Get the folder name from the request data
    folder_name = request.form.get('folder_name')
    haar_cascade = cv2.CascadeClassifier(classifier)
    (im_width, im_height) = (112, 92)
    if folder_name:
        # Create a folder if it doesn't exist
        folder_path = os.path.join(image_dir, folder_name)
        if not os.path.exists(folder_path):
            os.mkdir(folder_path)

        # Process each image from the request
        files = request.files.getlist('images')
        count = 0  # Counter for captured images
        for file in files:
            # Read image from file storage
            image_bytes = file.read()
            nparr = np.frombuffer(image_bytes, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            # Mao ning image shape ug size
            height, width, channels = frame.shape
            frame = cv2.flip(frame, 1)  # Flip frame
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert frame to grayscale

            # Scale down for speed
            size = 2  # Change this value as needed
            mini = cv2.resize(gray, (int(gray.shape[1] / size), int(gray.shape[0] / size)))

            # Detect faces
            faces = haar_cascade.detectMultiScale(mini)

            # We only consider largest face
            faces = sorted(faces, key=lambda x: x[3])
            if faces:
                face_i = faces[0]
                (x, y, w, h) = [v * size for v in face_i]

                face = gray[y:y + h, x:x + w]
                face_resize = cv2.resize(face, (im_width, im_height))

                # Draw rectangle and write name
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
                cv2.putText(frame, folder_name, (x - 10, y - 10), cv2.FONT_HERSHEY_PLAIN,
                            1, (0, 255, 255), 2)

                # Remove false positives
                if w * 6 < width or h * 6 < height:
                    print("Face too small")
                else:
                    # To create diversity, only save every fifth detected image
                    if count % 5 == 0:
                        filename = f"{count // 5 + 1}.png"  # Naming as 1.png, 2.png, ...
                        cv2.imwrite(os.path.join(folder_path, filename), face_resize)
                        print(f"Saving training sample {count // 5 + 1}")
                count += 1

        return jsonify({'message': f'Images uploaded to folder "{folder_name}" successfully'}), 201
    else:
        return jsonify({'error': 'Folder name not provided in the request'}), 400
# def upload_images():
#     # Get the folder name from the request data
#     folder_name = request.form.get('folder_name')

#     if folder_name:
#         # Create a folder if it doesn't exist
#         folder_path = os.path.join(image_dir, folder_name)
#         if not os.path.exists(folder_path):
#             os.mkdir(folder_path)

        
#         # Save uploaded images to the folder
#         for file in request.files.getlist('images'):
#             file.save(os.path.join(folder_path, file.filename))

#         return jsonify({'message': f'Images uploaded to folder "{folder_name}" successfully'}), 201
#     else:
#         return jsonify({'error': 'Folder name not provided in the request'}), 400

@app.route('/runscript')
def run_script():
    # Check if an image file is provided in the request
    try:
        subprocess.run(['python' , 'face_recog.py'], check=True)
        return 'Script Executed Successfully'
    
    except subprocess.CalledProcessError as e:
        return f"Error running script: {e}"


    # Response
@app.route('/facedetect', methods=['POST'])
def face_detect():
    print("Face Recognition Starting ...")
    # Create a list of images, labels, dictionary of corresponding names
    (images, labels, names, id) = ([], [], {}, 0)
    result = ''
    try:
    # Get the folders containing the training data
        for (subdirs, dirs, files) in os.walk(image_dir):
            # Loop through each folder named after the subject in the photos
            for subdir in dirs:
                names[id] = subdir
                subjectpath = os.path.join(image_dir, subdir)
                # Loop through each photo in the folder
                for filename in os.listdir(subjectpath):
                    # Skip non-image formats
                    f_name, f_extension = os.path.splitext(filename)
                    if(f_extension.lower() not in ['.png','.jpg','.jpeg','.gif','.pgm']):
                        print("Skipping "+filename+", wrong file type")
                        continue
                    path = subjectpath + '/' + filename
                    label = id
                    # Add to training data
                    images.append(cv2.imread(path, 0))
                    labels.append(int(label))
                id += 1

        (im_width, im_height) = (120, 102)

        # Create a Numpy array from the two lists above
        (images, labels) = [np.array(lis) for lis in [images, labels]]
        face_recognizer = cv2.face.LBPHFaceRecognizer_create()
        face_recognizer.train(images, labels)
        haar_cascade = cv2.CascadeClassifier(classifier)

        # Convert image to numpy array
        nparr = np.frombuffer(request.files['image'].read(), np.uint8)
        # Decode image
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)


        # Convert image to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Resize to speed up detection
        size = 2
        mini = cv2.resize(gray, (int(gray.shape[1] / size), int(gray.shape[0] / size)))

        # Detect faces
        faces = haar_cascade.detectMultiScale(mini)
        for i in range(len(faces)):
            face_i = faces[i]
            # Coordinates of face after scaling back by size
            (x, y, w, h) = [v * size for v in face_i]
            face = gray[y:y + h, x:x + w]
            face_resize = cv2.resize(face, (im_width, im_height))
            start =(x, y)
            end =(x + w, y + h)
            # Try to recognize the face
            prediction = face_recognizer.predict(face_resize)
            cv2.rectangle(image, start, end, (0, 255, 255), 3) # creating a bounding box for detected face
            cv2.rectangle(image, (start[0],start[1]-20), (start[0]+120,start[1]), (0, 255, 255), -3) # creating rectangle on the upper part of bounding box

            if prediction[1] < 90:
                cv2.putText(image, '%s - %.0f' % (names[prediction[0]], prediction[1]), (x + 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), thickness=2) #blue, green, red
                print('%s - %.0f' % (names[prediction[0]], prediction[1]))
                result = names[prediction[0]]
            else:
                result = "Unknown Person"
                cv2.putText(image, result, (x + 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), thickness=2)
                print("Unknown Person -", prediction[1])

        # Convert image to bytes for response
        _, img_encoded = cv2.imencode('.jpg', image)
        img_bytes = img_encoded.tobytes()

        # Response
        response = {'result': result}
        print(response)
        return jsonify(response), 200
    except KeyError:
        return jsonify({'error': 'Image file not found in request'}), 400

        
if __name__ == '__main__':
	app.run(debug=True, host='0.0.0.0')
	
