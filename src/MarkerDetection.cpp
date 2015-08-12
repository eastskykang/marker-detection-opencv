/*
 * MarkerDetection.cpp
 *
 *  Created on: Aug 5, 2015
 *      Author: Dongho Kang
 *       Email: east0822@gmail.com
 */

#include "MarkerDetection.h"

CMarkerDetection::CMarkerDetection (float marker_width, float marker_height)
: _marker_width(marker_width), _marker_height(marker_height) // This is initialization list
{
	cvInitFont(&_font, CV_FONT_HERSHEY_SIMPLEX, .4, .4, 0, 1, 8);

	CvSize image_size = cvSize(640, 480);
	_calib.LoadCalibParams(image_size);
}

CMarkerDetection::~CMarkerDetection()
{
}

void CMarkerDetection::DrawMarkerRect(IplImage *img, sMarkerInfo &mi, CvScalar color)
{
	CvPoint corner[4]; // For a rectangle

	for (int i=0; i<4; ++i) {
		corner[i] = cvPointFrom32f(mi.corner[i]);
	}

	cvLine (img, corner[0], corner[1], color, 2);
	cvLine (img, corner[1], corner[2], color, 2);
	cvLine (img, corner[2], corner[3], color, 2);
	cvLine (img, corner[3], corner[0], color, 2);

	cvLine (img, corner[0], corner[2], color, 2);
	cvLine (img, corner[1], corner[3], color, 2);
}

void CMarkerDetection::MarkerRecog(IplImage *src, IplImage *dst)
{
	//PERFORM("MarkerRecog()");

	IplImage *img_gray = cvCreateImage (cvGetSize(src), IPL_DEPTH_8U, 1);
	IplImage *img_bin = cvCreateImage (cvGetSize(img_gray), IPL_DEPTH_8U, 1);

	////////////////////////////// this part sharpens the image.
	//double mask1[3][3] = { {1.0, -2.0, 1.0}, {-2.0, 5.0, -2.0}, {1.0, -2.0, 1.0} };
	//double mask2[3][3] = { {0.0, -1.0, 0.0}, {-1.0, 5.0, -1.0}, {0.0, -1.0, 0.0} };
	//kernel1 = cvMat(3, 3, CV_64FC1, mask1);
	//kernel2 = cvMat(3, 3, CV_64FC1, mask2);

// 	double mask3[3][3] = { {-1.0, -1.0, -1.0}, {-1.0, 9.0+1.0, -1.0}, {-1.0, -1.0, -1.0} };
// 	CvMat kernel3;// kernel2, kernel3;
// 	kernel3 = cvMat(3, 3, CV_64FC1, mask3);
// 	cvFilter2D(src, src, &kernel3);

	//IplImage *img_gray_1 = cvCreateImage (cvGetSize(src), IPL_DEPTH_16U, 1);
	//cvSobel(img_gray, img_gray, 1,1,3);
	//cvShowImage("B", src);
	//////////////////////////////


	// 입력이미지를 gray 이미지로 바꾼다.
	cvCvtColor(src, img_gray, CV_RGB2GRAY);

	// 노이즈를 제거하기 위하여 가우시안 커널을 적용하여 이미지를 부드럽게 만든다.
	//cvSmooth (img_gray, img_gray, CV_GAUSSIAN, 3, 3);

	// 모폴로지 연산을 수행하여 이미지의 열린 부분을 없앤다.
	IplConvKernel *kernel = cvCreateStructuringElementEx (3, 3, 1, 1, CV_SHAPE_ELLIPSE, NULL);
	cvMorphologyEx(img_gray, img_gray, NULL, kernel, CV_MOP_CLOSE, 1);

	cvReleaseStructuringElement (&kernel);

	// gray 이미지를 복사하여 threshold값을 기준으로 binary 이미지로 만든다.
	cvCopyImage (img_gray, img_bin);

	// 두 함수 중 하나를 골라 쓴다.
	cvAdaptiveThreshold(img_bin,  img_bin, 255, CV_ADAPTIVE_THRESH_MEAN_C, CV_THRESH_BINARY, 31, 15);
	//cvThreshold (img_bin, img_bin, 63, 255, CV_THRESH_BINARY /* | CV_THRESH_OTSU*/);
	//cvShowImage("B", img_bin);
	// 컨투어를 찾아 저장할 메모리 공간 할당
	CvMemStorage *storage = cvCreateMemStorage(0);
	CvSeq *contours = NULL;

	int noContour = cvFindContours (img_bin, storage, &contours, sizeof(CvContour), CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE);
	// int noContour = cvFindContours (img_bin, storage, &contours, sizeof(CvContour), CV_RETR_LIST, CV_CHAIN_APPROX_NONE);

	if (0 < noContour) {
		CvSeq *approxContours = cvApproxPoly(contours, sizeof(CvContour), storage, CV_POLY_APPROX_DP, 1., 1);
		// cvDrawContours (dst, approxContours, CV_RGB(255,255,0), CV_RGB(0,255,0), 10, 1, CV_AA);

		_markers.clear ();
		FindMarkerInContour (approxContours, storage, 0); //User defined function not opencv function
		GetMarkerCode (img_gray, dst);

		vector<sMarkerInfo> markers_tmp;
		for (unsigned int i=0; i<_markers.size(); i++) {
			if (0 <= _markers[i].ID) {
				markers_tmp.push_back (_markers[i]);
			}
		}
		_markers.swap (markers_tmp);

		//for (unsigned int i=0; i<_markers.size(); ++i) {
		//	DrawMarkerRect(dst, _markers[i], CV_RGB (255, 0, 0));
		//}
	}

	cvReleaseMemStorage (&storage);
	cvReleaseImage(&img_bin);
	cvReleaseImage(&img_gray);
}

inline double Distance (CvPoint &p1, CvPoint &p2)
{
	// 두 점 p1, p2간의 유클리드 거리를 계산한다.

	double dx = p1.x - p2.x;
	double dy = p1.y - p2.y;

	return sqrt (dx*dx + dy*dy);
}

bool CMarkerDetection::CheckRectCenter(CvSeq *seq)
{
	CvPoint corner[4] = {
		*(CvPoint *)cvGetSeqElem(seq, 0),
		*(CvPoint *)cvGetSeqElem(seq, 1),
		*(CvPoint *)cvGetSeqElem(seq, 2),
		*(CvPoint *)cvGetSeqElem(seq, 3),
	};

	// 이 사각형의 두 대각선 교점이 사각형 중앙에 오는지 검사한다.
	// 여기서는 두 라인 l1, l2의 교점에서 t와 u 값을 계산하는 식이다.
	// l1 = (a1,b1) + t*(x1,y1)
	// l2 = (a2,b2) + u*(x2,y2)

	double a1 = corner[0].x;
	double b1 = corner[0].y;
	double x1 = corner[2].x - corner[0].x;
	double y1 = corner[2].y - corner[0].y;

	double a2 = corner[1].x;
	double b2 = corner[1].y;
	double x2 = corner[3].x - corner[1].x;
	double y2 = corner[3].y - corner[1].y;

	// 관련 식의 유도는 문서 참조: 직선의 방정식과 교점.docx
	CvMat *A = cvCreateMat (2, 2, CV_64FC1);
	CvMat *B = cvCreateMat (2, 1, CV_64FC1);
	CvMat *Ainv = cvCreateMat(2, 2, CV_64FC1);
	CvMat *x = cvCreateMat(2, 1, CV_64FC1);

	cvmSet (A, 0, 0,  2*(x1*x1 + y1*y1));
	cvmSet (A, 0, 1, -2*(x1*x2 + y1*y2));
	cvmSet (A, 1, 0, -2*(x1*x2 + y1*y2));
	cvmSet (A, 1, 1,  2*(x2*x2 + y2*y2));

	cvmSet (B, 0, 0,  2*(x1*(a2 - a1) + y1*(b2 - b1)));
	cvmSet (B, 1, 0, -2*(x2*(a2 - a1) + y2*(b2 - b1)));

	cvInvert (A, Ainv);
	cvMatMul (Ainv, B, x);

	double x00 = cvmGet (x, 0, 0);
	double x10 = cvmGet (x, 1, 0);

	cvReleaseMat (&A);
	cvReleaseMat (&B);
	cvReleaseMat (&Ainv);
	cvReleaseMat (&x);

	const double l_th = 0.15;
	if (fabs(x00 - 0.5) < l_th && fabs(x10 - 0.5) < l_th) {
		// 성공
		return true;
	}
	else return false;

	/*
	MatrixXd A(2,2);
    A(0,0) =  2*(x1*x1 + y1*y1);
    A(0,1) = -2*(x1*x2 + y1*y2);
    A(1,0) = -2*(x1*x2 + y1*y2);
    A(1,1) =  2*(x2*x2 + y2*y2);

 	MatrixXd B(2,1);
	B(0,0) =  2*(x1*(a2 - a1) + y1*(b2 - b1));
	B(1,0) = -2*(x2*(a2 - a1) + y2*(b2 - b1));

	// t, u 계산
	MatrixXd x = A.inverse()*B;

	// t, u 값이 0.5를 기준으로 +-0.15 사이에 오는지 검사한다.
	const double l_th = 0.15;
	if (fabs(x(0,0) - 0.5) < l_th && fabs(x(1,0) - 0.5) < l_th) {
		// 성공
		return true;
	}
	else return false;
	*/
}

void CMarkerDetection::FindMarkerInContour (CvSeq *contours, CvMemStorage *storage, int level)
{
	for (CvSeq *s = contours; s; s = s->h_next){
		// 컨투어를 구성하는 점의 수가 4개 이상 되어야 사각형 후보가 된다.
		if (s->total >= 4) {
			// 바운딩 박스를 찾는 이유는 컨투어의 대략적인 크기를 알기 위해서다.
			// 크기에 따라 컨투어를 approximation 하는 정밀도를 조정한다.
			// 여기서는 대략 10%정도의 정밀도로 조정한다. (d*approx_param 부분)
			CvRect rect = cvBoundingRect (s);

			double d = sqrt ((double)rect.height*rect.width);

			const double d_th = 12.;
			const double approx_param = 0.1;

			// 컨투어의 대략적인 크기가 d_th보다 커야 한다.
			if (d > d_th) {
				CvSeq *ss = cvApproxPoly (s, s->header_size, storage, CV_POLY_APPROX_DP, d*approx_param, 0);
				// 컨투어를 approximation 하고나서 코너의 수가 4개(사각형)인지 검사한다.
				if (ss->total == 4) {
					// 추가적으로, 이 사각형의 두 대각선 교점이 사각형 중앙에 오는지 검사한다.
					if (CheckRectCenter(ss)) {
						// 마커를 찾았다. 마커 목록에 저장한다.
						sMarkerInfo mi;

						mi.level = level;
						mi.width = _marker_width;		// 실제 마커의 가로 길이 (단위: m)
						mi.height = _marker_height;		// 실제 마커의 세로 길이 (단위: m)
						mi.ID = -1;						// -1로 초기화
						mi.corner[0] = cvPointTo32f(*(CvPoint *)cvGetSeqElem(ss, 0));
						mi.corner[1] = cvPointTo32f(*(CvPoint *)cvGetSeqElem(ss, 1));
						mi.corner[2] = cvPointTo32f(*(CvPoint *)cvGetSeqElem(ss, 2));
						mi.corner[3] = cvPointTo32f(*(CvPoint *)cvGetSeqElem(ss, 3));

						_markers.push_back (mi);
					}
				}
			}
		}

		if (s->v_next) {
			FindMarkerInContour (s->v_next, storage, level+1);
		}
	}
}

void CMarkerDetection::GetMarkerCode(IplImage *src, IplImage *dst)
{
	for (unsigned int i=0; i<_markers.size(); ++i) {
		// 검출된 마커의 코너로부터 서브픽셀 정확도로 코너 좌표를 다시 구한다.
		cvFindCornerSubPix (src, _markers[i].corner, 4, cvSize(2, 2), cvSize(-1, -1),
			cvTermCriteria(CV_TERMCRIT_EPS+CV_TERMCRIT_ITER, 10, 0.01));

		// src에서 찾은 마커의 영역으로부터 마커 영역만 추출한다.
		const int marker_size = 60;
		IplImage *img_marker = cvCreateImage (cvSize (marker_size, marker_size), IPL_DEPTH_8U, 1);
		ExtractMarkerImage (src, img_marker, _markers[i]);

		// 마커는 6 x 6의 행렬로 구성된다.
		double code_matrix[6][6] = {0, };

		// 마커 내부의 픽셀들의 합으로부터 코드 값을 추출한다.
		ExtractCodeFromImage (img_marker, code_matrix);

		if (CheckParity (code_matrix)) {
			int rotate_index = GetRotation (code_matrix);
			if (0 <= rotate_index) {
				// 마커 인식 성공!!!

				// 마커의 코드를 포함한 행렬의 회전된 각도를 보정해 준다.
				RotateMatrix (code_matrix, rotate_index);
				RotateCorner (_markers[i].corner, rotate_index, _markers[i].level%2);

				_markers[i].ID = CalcMarkerID (code_matrix);
				// TRACE ("Marker ID = %d\n", _markers[i].ID);

				FindMarkerPos3d (&_markers[i]);

				DrawMarkerInfo (&_markers[i], dst);

				// 원본 마커 코드
				 //cvNamedWindow ("Marker Image Org", CV_WINDOW_AUTOSIZE);
				 //cvShowImage ("Marker Image Org", img_marker);

				 //ShowMarkerCode (cvGetSize(img_marker), code_matrix);
			}
		}

		cvReleaseImage (&img_marker);
	}
}

void CMarkerDetection::ExtractMarkerImage (IplImage *src, IplImage *dst, sMarkerInfo &mi)
{
	assert (src->nChannels == 1);
	assert (dst->width == dst->height);

	const float ignoring_margin = 0.f;	// 원본 이미지로부터 마커 이미지로 복사하면서 무시할 테두리의 영역

	CvMat *transform_matrix = cvCreateMat(3, 3, CV_32FC1);

	if (mi.level%2 == 0) {
		// 추출한 마커를 저장할 이미지 상의 좌표
		CvPoint2D32f dest_corner_cw[4] = {
			{ -ignoring_margin,				-ignoring_margin},
			{ -ignoring_margin,				dst->height + ignoring_margin},
			{ dst->width + ignoring_margin,	dst->height + ignoring_margin},
			{ dst->width + ignoring_margin,	-ignoring_margin},
		};

		// 소스 이미지에서 마커의 코너에 대한 점들을 마커 이미지 상의 점들로 매핑하기 위한 변환 행렬을 구한다.
		cvGetPerspectiveTransform (mi.corner, dest_corner_cw, transform_matrix);
	}
	else {
		CvPoint2D32f dest_corner_ccw[4] = {
			{ dst->width + ignoring_margin,	-ignoring_margin},
			{ dst->width + ignoring_margin,	dst->height + ignoring_margin},
			{ -ignoring_margin,				dst->height + ignoring_margin},
			{ -ignoring_margin,				-ignoring_margin},
		};

		// 소스 이미지에서 마커의 코너에 대한 점들을 마커 이미지 상의 점들로 매핑하기 위한 변환 행렬을 구한다.
		cvGetPerspectiveTransform (mi.corner, dest_corner_ccw, transform_matrix);
	}

	// 소스 이미지 상의 마커를 마커 이미지로 복사한다.
	cvWarpPerspective (src, dst, transform_matrix);

	if (mi.level%2 == 0) {
		cvNot (dst, dst);
	}

	cvReleaseMat (&transform_matrix);
}

void CMarkerDetection::ExtractCodeFromImage (IplImage *src, double code_matrix[6][6])
{
	#define PIXEL_YX(img,y,x)	(unsigned char &)img->imageData[(y)*img->widthStep + (x)]

	assert (src->width == 60 && src->height == 60);

	// 마커 이미지를 6x6 격자로 쪼갠 후 각각의 격자 내부 픽셀들을 모두 더한다.
	for (int y = 0; y < src->height; y++) {
		for (int x = 0; x < src->width; x++) {
			int yi = y/10;
			int xi = x/10;

			assert (yi < 6 && xi < 6);
			code_matrix[yi][xi] += PIXEL_YX(src, y, x);
		}
	}

	double min_v = 255.;
	double max_v = 0.;

	// 더한 값을 0 ~ 1 사이의 값으로 정규화 하면서 최대값과 최소값을 찾는다.
	// 하나의 격자에는 100개의 픽셀이 더해지고 한 픽셀의 최대 값은 255이기 때문에
	// 더한 값을 100*255로 나누어주면 된다.
	for (int y = 0; y < 6; y++) {
		for (int x = 0; x < 6; x++) {
			code_matrix[y][x] /= 100.*255;

			if (min_v > code_matrix[y][x]) min_v = code_matrix[y][x];
			if (max_v < code_matrix[y][x]) max_v = code_matrix[y][x];
		}
	}

	// 최대값과 최소값의 중간값을 찾는다.
	double mid_v = (min_v + max_v)/2.;

	// 중간값을 기준으로 검정색에 가까우면 1.을 흰색에 가까우면 0.을 대입한다.
	for (int y = 0; y < 6; y++) {
		for (int x = 0; x < 6; x++) {
			code_matrix[y][x] = (code_matrix[y][x] < mid_v) ? 1. : 0.;
		}
	}
}

bool CMarkerDetection::CheckParity (double code_matrix[6][6])
{
	int sum = 0;

	// 테두리가 모두 제대로 있는지 검사한다.
	// 즉, 한 방향의 블럭 수는 6개이고 모서리가 4개이니까
	// 합이 24개가 되어야 한다.
	for (int i = 0; i < 6; i++) {
		sum += (int)code_matrix[0][i];
		sum += (int)code_matrix[5][i];
		sum += (int)code_matrix[i][0];
		sum += (int)code_matrix[i][5];
	}
	if (sum != 24) return false;

	sum = 0;

	// 체크섬을 검사한다.
	// 테두리를 제외한 내부 블럭의 수는 짝수가 되어야 한다.
	for (int y = 1; y < 5; y++) {
		for (int x = 1; x < 5; x++) {
			sum += (int)code_matrix[y][x];
		}
	}
	return (sum%2 == 0);
}


int CMarkerDetection::GetRotation (double code_matrix[6][6])
{
	if      (code_matrix[1][1] && !code_matrix[1][4] && !code_matrix[4][4] && !code_matrix[4][1]) return 0;	// 정상
	else if (!code_matrix[1][1] && code_matrix[1][4] && !code_matrix[4][4] && !code_matrix[4][1]) return 1;	// 시계방향으로 90도 회전됨
	else if (!code_matrix[1][1] && !code_matrix[1][4] && code_matrix[4][4] && !code_matrix[4][1]) return 2; // 시계방향으로 180도 회전됨
	else if (!code_matrix[1][1] && !code_matrix[1][4] && !code_matrix[4][4] && code_matrix[4][1]) return 3; // 시계방향으로 270도 회전됨
	else return -1; // 있을수 없는 조합이다. 실패
}


void CMarkerDetection::RotateMatrix (double code_matrix[6][6], int angle_idx)
{
	if (angle_idx == 0) return;

	double cb[6][6];

	for (int y = 0; y < 6; y++) {
		for (int x = 0; x < 6; x++) {
			switch (angle_idx) {
			case 1: cb[y][x] = code_matrix[x][5-y];		break; // 반시계 방향으로 90도 회전
			case 2: cb[y][x] = code_matrix[5-y][5-x];	break; // 반시계 방향으로 180도 회전
			case 3: cb[y][x] = code_matrix[5-x][y];		break; // 반시계 방향으로 270도 회전
			}
		}
	}
	memcpy (code_matrix, cb, sizeof(double)*6*6);
}

inline void swap (CvPoint2D32f &c1, CvPoint2D32f &c2)
{
	CvPoint2D32f temp = c1;
	c1 = c2;
	c2 = temp;
}

void CMarkerDetection::RotateCorner (CvPoint2D32f corner[4], int angle_idx, int dir)
{
	CvPoint2D32f c[4];

	if (dir) {
		for (int i=0; i<4; ++i) {
			c[i] = corner[(i + 3 + angle_idx)%4];
		}
		swap (c[1], c[3]);
	}
	else {
		for (int i=0; i<4; ++i) {
			c[i] = corner[(i + 4 - angle_idx)%4];
		}
	}
	memcpy (corner, c, sizeof(CvPoint2D32f)*4);
}

int CMarkerDetection::CalcMarkerID (double code_matrix[6][6])
{
	int id = 0;
	if (code_matrix[4][2]) id += 1;
	if (code_matrix[3][4]) id += 2;
	if (code_matrix[3][3]) id += 4;
	if (code_matrix[3][2]) id += 8;
	if (code_matrix[3][1]) id += 16;
	if (code_matrix[2][4]) id += 32;
	if (code_matrix[2][3]) id += 64;
	if (code_matrix[2][2]) id += 128;
	if (code_matrix[2][1]) id += 256;
	if (code_matrix[1][3]) id += 512;
	if (code_matrix[1][2]) id += 1024;

	return id;
}

void CMarkerDetection::FindMarkerPos3d (sMarkerInfo *marker)
{
	if (_calib._intrinsic_matrix && _calib._distortion_coeffs)
	{

	} else return;

	// 회전(rotation)과 이동(translation)을 계산하여 저장할 매트릭스 생성
	CvMat rotation    = cvMat(3, 1, CV_32FC1, marker->rotation);
	CvMat translation = cvMat(3, 1, CV_32FC1, marker->translation);

	float image_xy[4][2] = {
		{ marker->corner[0].x, marker->corner[0].y },
		{ marker->corner[1].x, marker->corner[1].y },
		{ marker->corner[2].x, marker->corner[2].y },
		{ marker->corner[3].x, marker->corner[3].y },
	};

	float object_xyz[4][3] = {
		{ 0.0f,				0.0f,			0.0f },
		{ 0.0f,				marker->height,	0.0f },
		{ marker->width,	marker->height,	0.0f },
		{ marker->width,	0.0f,			0.0f },
	};

	CvMat object_points = cvMat(4, 3, CV_32FC1, &object_xyz[0][0]);
	CvMat image_points  = cvMat(4, 2, CV_32FC1, &image_xy[0][0]);

	// 3차원 공간에서 마커의 위치와 방위를 찾는다.
	cvFindExtrinsicCameraParams2 (&object_points, &image_points,
		_calib._intrinsic_matrix, _calib._distortion_coeffs,
		&rotation, &translation);
}

void CMarkerDetection::ShowMarkerCode (CvSize &size, double code_matrix[6][6])
{
	// 코드 블록으로부터 만들어낸 마커 코드를 이미지로 변환하여 표시한다.

	IplImage* img = cvCreateImage (size, IPL_DEPTH_8U, 1);

	cvSet (img, CV_RGB(255,255,255));

	double dx = img->width/6.;
	double dy = img->height/6.;

	for (int y = 0; y < 6; y++) {
		for (int x = 0; x < 6; x++) {
			if (code_matrix[y][x]) {
				cvDrawRect(img, cvPoint(cvRound(x*dx), cvRound(y*dy)),
					cvPoint (cvRound((x+1)*dx), cvRound((y+1)*dy)), CV_RGB(0,0,0), CV_FILLED);
			}
		}
	}

	cvNamedWindow ("Marker Image", CV_WINDOW_AUTOSIZE);
	cvShowImage ("Marker Image", img);

	cvReleaseImage(&img);
}

void CMarkerDetection::DrawMarkerInfo (sMarkerInfo *marker, IplImage *dst)
{
	float depth = max(marker->width, marker->height);

	// 3차원 공간에서 마커의 x, y, z 좌표를 설정한다.
	float object_xyz[4][3] = {
		{0.f,			0.f,			0.f		},
		{marker->width,	0.f,			0.f		},
		{0.f,			marker->height,	0.f		},
		{0.f,			0.f,			depth	},
	};
	float image_xy[4][2] = { 0.f, };

	CvMat image_points  = cvMat(4, 2, CV_32FC1, &image_xy[0][0]);
	CvMat object_points = cvMat(4, 3, CV_32FC1, &object_xyz[0][0]);

	CvMat rotation    = cvMat(3, 1, CV_32FC1, marker->rotation);
	CvMat translation = cvMat(3, 1, CV_32FC1, marker->translation);

	// 마커의 x, y, z 좌표를 이미지로 프로젝션 한다.
	cvProjectPoints2 (&object_points, &rotation, &translation, _calib._intrinsic_matrix, _calib._distortion_coeffs, &image_points);

	// 2차원으로 프로젝션된 좌표를 그린다.
	cvLine (dst, cvPoint(cvRound(image_xy[0][0]), cvRound(image_xy[0][1])), cvPoint(cvRound(image_xy[1][0]), cvRound(image_xy[1][1])), CV_RGB(255,0,0), 2);
	cvLine (dst, cvPoint(cvRound(image_xy[0][0]), cvRound(image_xy[0][1])), cvPoint(cvRound(image_xy[2][0]), cvRound(image_xy[2][1])), CV_RGB(0,255,0), 2);
	//cvLine (dst, cvPoint(cvRound(image_xy[0][0]), cvRound(image_xy[0][1])), cvPoint(cvRound(image_xy[3][0]), cvRound(image_xy[3][1])), CV_RGB(0,0,255), 2);

	// 마커의 ID를 표시한다.
	char buff[256];
	sprintf (buff, "     ID: %d", marker->ID);
	cvPutText(dst, buff, cvPointFrom32f(marker->corner[0]), &_font, CV_RGB(255, 0, 0));
}
