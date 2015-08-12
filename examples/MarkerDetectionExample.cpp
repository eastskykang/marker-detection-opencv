/*
 * MarkerDetectionExample.cpp
 *
 *  Created on: Aug 5, 2015
 *      Author: Dongho Kang
 *       Email: east0822@gmail.com
 */

#include <opencv/cv.h>
#include <opencv/cxcore.h>
#include <opencv/highgui.h>
#include <opencv2/opencv.hpp>
#include <opencv2/legacy/legacy.hpp>

#include "MarkerDetection.h"

CMarkerDetection *_markRecog;
CvCapture* _capture;

void PrintMarkerInfo(const vector<sMarkerInfo> &markers);

int main() {

	_capture = cvCreateCameraCapture(0);
	cvSetCaptureProperty(_capture, CV_CAP_PROP_FRAME_WIDTH, 640.0);
	cvSetCaptureProperty(_capture, CV_CAP_PROP_FRAME_HEIGHT, 480.0);
	cvSetCaptureProperty(_capture, CV_CAP_PROP_FPS, 30.0);

	_markRecog = new CMarkerDetection(0.128, 0.128);

	IplImage *frame = NULL;
	frame = cvQueryFrame(_capture);

	if (frame == NULL) {
		std::cout << "camera has not been initialized" << std::endl;
		return -1;
	}

	// cam windows
	cvNamedWindow("cam");

	while (true) {
		// capture camera image
		frame = cvQueryFrame(_capture);

		IplImage *und = cvCreateImage(cvGetSize(frame), IPL_DEPTH_8U, 3);	// undistorted image
		IplImage *fin = cvCreateImage(cvGetSize(frame), IPL_DEPTH_8U, 3);	// final image (marker detection)

		// need to be calibrated
		_markRecog->_calib.Undistort(frame, und);
		cvCopy(und, fin, 0);
		_markRecog->MarkerRecog(und, fin);

		// show marker axis and id (if you need it)
		PrintMarkerInfo(_markRecog->_markers);

		// show the captured image
		cvShowImage("cam", fin);

		cvWaitKey(1);

		cvReleaseImage(&dst);
		cvReleaseImage(&und);
	}

	// release all
	cvReleaseCapture(&_capture);
	cvReleaseImage(&frame);
	cvDestroyAllWindows();
	return 0;
}

void PrintMarkerInfo (const vector<sMarkerInfo> &markers)
{
	// function for printing marker's information
	for (unsigned int i=0; i<markers.size(); ++i) {
		const sMarkerInfo &mi = markers[i];

		std::cout << "Marker ID          = " << (int)mi.ID << std::endl;
		std::cout << "Rotation Vector    = " << (double)mi.rotation[0] << ", " <<  (double)mi.rotation[1] <<", " << (double)mi.rotation[2] << std::endl;
		std::cout << "Translation Vector = " << (double)mi.translation[0] << ", " << (double)mi.translation[1] << ", " << (double)mi.translation[2] << std::endl;
	}
}
