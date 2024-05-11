#AUTHOR: Lloyd E. Agonia
#COURSE & YEAR:BSEMT-MR3P1
import cv2, sys, numpy, os,time
import tkinter as tk
from tkinter import messagebox

size = 2 # change this to 4 to speed up processing trade off is the accuracy
classifier = 'haarcascade_frontalface_default.xml'
image_dir = 'images'

#Main window sa GUI
window=tk.Tk()
window.title("Car Face Detection")
window.config(background="green")


#Para sa Name labeling ug Input
l5=tk.Label(window,text="       " ,font=("Times New Roman" ,20),bg='green',fg='black')
l5.grid(column=0, row=5)
l1=tk.Label(window,text=" USERNAME" ,font=("Times New Roman" ,50),bg='green',fg='black')
l1.grid(column=0, row=10)
t1=tk.Entry(window,width=18,bd=3,font=("Times New Roman" ,30),fg='blue') #Box para sa pangalan
t1.grid(column=1, row=10)

def register():
    
    if(t1.get()==""):
        messagebox.showinfo('NOTE','Please provide a name first!')
        
    else:
        try:
            name_class = t1.get()# Kuhaon niya ang text sa Entry box then mao iya i-name sa folder per name-class
        except:
            print("Please provide a name first!")
            sys.exit(0)
        path = os.path.join(image_dir, name_class)
        if not os.path.isdir(path):
            os.mkdir(path)
        (im_width, im_height) = (112, 92) #yoooooooooooooooooooooooooooooooooooooooooow
        haar_cascade = cv2.CascadeClassifier(classifier)
        webcam = cv2.VideoCapture(0)

        # Generate name for image file
        pin=sorted([int(n[:n.find('.')]) for n in os.listdir(path)
            if n[0]!='.' ]+[0])[-1] + 1

        # Beginning message
        print("\n\033[94mThe program will save 30 samples. \
        Move your head around to increase accuracy.\033[0m\n")

        # The program loops until it has 50 images of the face.
        # Pwede dakoon ang sample size para mas accurate pero mas lag siya. 
        count = 0
        pause = 0
        count_max = 30   # desired number of sample per class
        while count < count_max:

            # Loop until the camera is working
            rval = False
            while(not rval):
                # Put the image from the webcam into 'frame'
                (rval, frame) = webcam.read()
                if(not rval):
                    print("Failed to open webcam. Trying again...")

            # Mao ning image shape ug size.
            height, width, channels = frame.shape # 640 , 480 ,3

            # Flip frame
            frame = cv2.flip(frame, 1)

            # Himoon grayscale ang tibook frame 
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Scale down for speed 
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
                cv2.putText(frame, name_class, (x - 10, y - 10), cv2.FONT_HERSHEY_PLAIN,
                    1,(0, 255, 255),2)

                # Remove false positives
                if(w * 6 < width or h * 6 < height):
                    print("Face too small")
                else:

                    # To create diversity, only save every fith detected image
                    if(pause == 0):

                        print("Saving training sample "+str(count+1)+"/"+str(count_max))

                        # Save image file
                        cv2.imwrite('%s/%s.png' % (path, pin), face_resize)

                        pin += 1
                        count += 1

                        pause = 1

            if(pause > 0):
                pause = (pause + 1) % 5
            cv2.imshow('Lingi Wala, Tuo Gwapo/Gwapa!', frame)
            if cv2.waitKey(1) & 0xFF == ord('q') or int(count)==30:
                print("Sampling Complete!")
                cv2.destroyAllWindows()
                break
b1=tk.Button(window,text="Register Face", font=("Times New Roman" ,50),bg='green',fg='yellow',command=register)
b1.place(x=390, y=200,anchor="center")

def face_detect():

    print("Face Recognition Starting ...")
    # Create a list of images,labels,dictionary of corresponding names
    (images, labels, names, id) = ([], [], {}, 0)

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
                if(f_extension.lower() not in
                        ['.png','.jpg','.jpeg','.gif','.pgm']):
                    print("Skipping "+filename+", wrong file type")
                    continue
                path = subjectpath + '/' + filename
                label = id

                # Add to training data
                images.append(cv2.imread(path, 0))
                labels.append(int(label))
            id += 1
    (im_width, im_height) = (120, 102) #yooooooooooooooooooow

    # Create a Numpy array from the two lists above
    (images, labels) = [numpy.array(lis) for lis in [images, labels]]
    face_recognizer = cv2.face.LBPHFaceRecognizer_create()
    face_recognizer.train(images, labels)
    haar_cascade = cv2.CascadeClassifier(classifier)
    webcam = cv2.VideoCapture(0) #  0 to use webcam 
    while True:
        # Loop until the camera is working
        rval = False
        while(not rval):
            # Put the image from the webcam into 'frame'
            (rval, frame) = webcam.read()
            if(not rval):
                print("Failed to open webcam. Trying again...")
        startTime = time.time()
        # Flip the image (optional)
        frame=cv2.flip(frame, 1) # 0 = horizontal ,1 = vertical , -1 = both

        # Convert to grayscalel
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Resize to speed up detection (optinal, change size above)
        mini = cv2.resize(gray, (int(gray.shape[1] / size), int(gray.shape[0] / size)))

        # Detect faces and loop through each one
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
            cv2.rectangle(frame,start , end, (0, 255, 255), 3) # creating a bounding box for detected face   #blue, green, red
            cv2.rectangle(frame, (start[0],start[1]-20), (start[0]+120,start[1]), (0, 255, 255), -3) # creating  rectangle on the upper part of bounding box
            #for i in prediction[1]
            if prediction[1]<90 :  # note: 0 is the perfect match  the higher the value the lower the accuracy
                cv2.putText(frame,'%s - %.0f' % (names[prediction[0]],prediction[1]),(x+5, y-5), cv2.FONT_HERSHEY_SIMPLEX,0.6,(0, 0, 255),thickness=2) #blue, green, red
                print('%s - %.0f' % (names[prediction[0]],prediction[1]))
            else:
                cv2.putText(frame,("Unknown Person {} ".format(str(int(prediction[1])))),(x+5, y-5), cv2.FONT_HERSHEY_SIMPLEX,0.5,(0, 0, 255),thickness=2)
                print("Unknown Person -",prediction[1])
        endTime = time.time()  
        try:
            fps = 1/(endTime-startTime)
        except:
            fps = 60  
        cv2.rectangle(frame,(30,48),(130,70),(0,0,255),-1)
        cv2.putText(frame,"FPS : {} ".format(str(int(fps))),(34,65),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,255),2)
        cv2.putText(frame,"Press Q to Exit",(240,470),cv2.FONT_HERSHEY_TRIPLEX,0.6,(255,0,0),1)
        # Show the image and check for "q" being pressed
        cv2.imshow('I SEE YOU  ', frame)
        

        if (cv2.waitKey(10) == ord('q')):
                    
                    cv2.destroyAllWindows()
                    break
b2=tk.Button(window,text="  Detect Face ", font=("Times New Roman" ,50),bg='green',fg='yellow',command=face_detect)
b2.place(x=390, y=300,anchor="center")


def exit():    
    print('Yeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeet')
    window.destroy()


b4=tk.Button(window,text="      EXIT       ", font=("Times New Roman" ,50),bg='green',fg='yellow',command=exit)
b4.place(x=390, y=400,anchor="center")


#Lenght and width sa GUI
window.geometry("800x600")
window.mainloop()