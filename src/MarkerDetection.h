/*
 * MarkerDetection.h
 *
 *  Created on: Aug 5, 2015
 *      Author: Dongho Kang
 *       Email: east0822@gmail.com
 */

#ifndef SRC_MARKERDETECTION_H_
#define SRC_MARKERDETECTION_H_

#include <vector>
#include <opencv/cv.h>
#include <opencv/cxcore.h>
#include <opencv/highgui.h>
#include <opencv2/opencv.hpp>
#include <opencv2/legacy/legacy.hpp>
#include "CamCalib.h"

using namespace std;

struct sMarkerInfo
{
	int level;
	float width, height;	// width / height of marker
	CvPoint2D32f center;	// center of marker
	CvPoint2D32f corner[4];	// corner of marker
	int ID;					// ID of marker

	float rotation[3];		// rotation vector of marker (3 x 1)
	float translation[3];	// translation vector of marker (3 x 1)
};

class CMarkerDetection
{
public:
	CMarkerDetection (float marker_width, float marker_height);
	~CMarkerDetection();

	void DrawMarkerRect(IplImage *src, sMarkerInfo &mi, CvScalar color);
	void MarkerRecog(IplImage *src, IplImage *dst);

public:
	CCamCalib _calib;
	vector<sMarkerInfo> _markers;
	float _marker_width;
	float _marker_height;

private:
	void FindMarkerInContour (CvSeq *contours, CvMemStorage *storage, int level);
	bool CheckRectCenter(CvSeq *seq);

	void GetMarkerCode(IplImage *src, IplImage *dst);
	void ExtractMarkerImage (IplImage *src, IplImage *dst, sMarkerInfo &mi);
	void ExtractCodeFromImage (IplImage *src, double code_matrix[6][6]);

	bool CheckParity (double code_matrix[6][6]);
	int  GetRotation (double code_matrix[6][6]);
	void RotateMatrix (double code_matrix[6][6], int rotate_index);
	void RotateCorner (CvPoint2D32f corner[4], int angle_idx, int dir);
	int  CalcMarkerID (double code_matrix[6][6]);

	void FindMarkerPos3d (sMarkerInfo *marker);

	void DrawMarkerInfo (sMarkerInfo *marker, IplImage *dst);
	void ShowMarkerCode (CvSize &size, double code_matrix[6][6]);

private:
	CvFont _font;
};

#endif /* SRC_MARKERDETECTION_H_ */
