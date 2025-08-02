// C++ program for the above approach
#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/objdetect.hpp>

using namespace cv;
using namespace std;

int main(int argc, char** argv) {
    VideoCapture cap(0, CAP_V4L2);
    if(!cap.isOpened()) {
        cerr << "Error: Cannot open camera.\n" ;
    }

    cap.set(CAP_PROP_FRAME_WIDTH, 640);
    cap.set(CAP_PROP_FRAME_HEIGHT, 480);
    cap.set(CAP_PROP_FPS, 60);

    int width = static_cast<int>(cap.get(CAP_PROP_FRAME_WIDTH)) ;
    int height = static_cast<int>(cap.get(CAP_PROP_FRAME_HEIGHT)) ;
    int fps = static_cast<int>(cap.get(CAP_PROP_FPS)) ;

    cout << "camera width: " << width << endl
         << "camera height: " << height << endl
         << "camera fps: " << fps << endl;

    // load the haarcascades classifiers
    CascadeClassifier faceCascade, eyeCascade, smileCascade;
    if(!faceCascade.load("/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml") ||
        !eyeCascade.load("/usr/share/opencv4/haarcascades/haarcascade_eye.xml") ||
        !smileCascade.load("/usr/share/opencv4/haarcascades/haarcascade_smile.xml")) {
        cout << "Error loading cascades" << endl;
        return -1;
    }

    Mat frame;
    while(true) {
        cap >> frame;
        if(frame.empty()) break;
        vector<Rect> faces;
        faceCascade.detectMultiScale(frame, faces, 1.1, 8, 0, Size(100, 100));

        for(const Rect& face: faces) {
            Rect upperFace(face.x, face.y, face.width, face.height / 2);
            Mat faceROI = frame(upperFace);
            Rect lowerFace(face.x, face.y + face.height / 2, face.width, face.height / 2);
            Mat faceROI2 = frame(lowerFace);
            vector<Rect> eyes;
            eyeCascade.detectMultiScale(faceROI, eyes, 1.2, 12, 0, Size(30, 30));
            for(const Rect& eye: eyes) {
                Rect eyeInFrame(face.x + eye.x, face.y + eye.y, eye.width, eye.height);
                rectangle(frame, eyeInFrame, Scalar(255, 255, 255), FILLED);
            }
            vector<Rect> smiles;
            smileCascade.detectMultiScale(faceROI2, smiles, 1.2, 25, 0, Size(70, 60));
            for(const Rect& smile: smiles) {
                Rect smileInFrame(face.x + smile.x, face.y + smile.y + face.height / 2, smile.width, smile.height);
                rectangle(frame, smileInFrame, Scalar(255, 0, 255), FILLED);
            }
            //debug
            rectangle(frame, face, Scalar(0, 255, 0), 2);
        }

        imshow("Live*", frame);
        switch(waitKey(1)) {
            case ' ': imwrite("screenshot.png", frame); break;
            case 27 : goto endwhile; break; //ESC to exit
        }
    }

    endwhile:
    cap.release();
    return 0;
}
