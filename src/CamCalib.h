/*
 * CamCalib.h
 *
 *  Created on: Aug 5, 2015
 *      Author: east0822
 */

#ifndef SRC_CAMCALIB_H_
#define SRC_CAMCALIB_H_

#include <opencv/cv.h>
#include <opencv/cxcore.h>
#include <opencv/highgui.h>
#include <opencv2/opencv.hpp>
#include <opencv2/legacy/legacy.hpp>

class CCamCalib
{
public:
	CCamCalib(int board_w = 9, int board_h = 6, int n_boards = 2, float cell_w = 0.035f, float cell_h = 0.035f);
	virtual ~CCamCalib();

	void LoadCalibParams (CvSize &image_size);
	bool FindChessboard(IplImage *src, IplImage *dst);
	void Undistort(IplImage *src, IplImage *dst);
	void CalibrateCamera(CvSize &image_size);

	CvMat* _image_points;
	CvMat* _object_points;
	CvMat* _point_counts;

	CvMat* _intrinsic_matrix;
	CvMat* _distortion_coeffs;

	IplImage* _mapx;
	IplImage* _mapy;

	float _cell_w;	// width of one block in chess board
	float _cell_h;	// height of one block in chess board

	int _n_boards;	// number of detection
	int _board_w;	// 체스판의 가로방향 코너 수
	int _board_h;	// 체스판의 세로방향 코너 수
	int _board_n;	// 가로 x 세로 방향의 코너 수
	int _board_total;
	int _successes;
};


#endif /* SRC_CAMCALIB_H_ */
