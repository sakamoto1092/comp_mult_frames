#include <cv.h>
#include <highgui.h>
#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include<opencv2/nonfree/nonfree.hpp>
#include<opencv2/nonfree/features2d.hpp>
#include <sstream>
#include <string>
#include <iostream>
#include <vector>
#include <fstream>
#include <boost/program_options.hpp>
#include "3dms-func.h"

// パノラマ画像の大きさ
#define PANO_W 6000
#define PANO_H 3000

// 合成フレームのFPS
#define TARGET_VIDEO_FPS 30

using namespace std;
using namespace cv;
using boost::program_options::options_description;
using boost::program_options::value;
using boost::program_options::variables_map;
using boost::program_options::store;
using boost::program_options::parse_command_line;
using boost::program_options::notify;

int main(int argc, char** argv) {

	//ここからフレーム合成プログラム

	VideoCapture frame_cap;							  // target frame movie
	VideoCapture pano_cap; 							  // background frame movie
	FileStorage cvfs("log.xml", CV_STORAGE_READ);  // パノラマ背景生成時のホモグラフィ行列のログファイル

	Mat panorama, target_frame, near_frame;
	Mat homography;

	ifstream ifs_target_cam;
	ifstream ifs_pano_cam;
	long s_pano; // panorama frame time

	//	string str_pano, str_frame, str_video;
	string str_pano_video, str_pano_time, str_pano_ori;
	string str_target_video, str_target_time, str_target_ori;
	string str_pano;
	Mat transform_image; // 画像単体での変換結果
	Mat transform_image2 = Mat(Size(PANO_W, PANO_H), CV_8UC3);

	vector<Mat> target_hist_channels;
	vector<Mat> near_hist_channels;

	Mat mask = Mat(Size(PANO_W, PANO_H), CV_8U, Scalar::all(0)); // パノラマ画像のマスク
	Mat pano_black = Mat(Size(PANO_W, PANO_H), CV_8U, Scalar::all(0)); // パノラマ画像と同じサイズの黒画像
	Mat white_img = Mat(Size(1280, 720), CV_8U, Scalar::all(255)); // フレームと同じサイズの白画像
	Mat gray_img1, gray_img2;
	Mat mask2;

	vector<Point2f> pt1, pt2;
	//	fstream fs(cam_data.txt);
	//	fs >> str_video;

	// 対応点の対の格納先
	std::vector<cv::DMatch> matches; // matcherにより求めたおおまかな対を格納

	// 特徴点の集合と特徴量
	std::vector<KeyPoint> objectKeypoints, imageKeypoints;
	Mat objectDescriptors, imageDescriptors;

	string algorithm_type("SURF");
	Ptr<Feature2D> feature;
	int hessian;

	int skip; // 合成開始フレーム番号
	long end; // 合成終了フレーム番号
	long frame_num; // 現在のフレーム位置
	int blur; // ブレのしきい値
	long FRAME_MAX; // 動画の最大フレーム数
	int FRAME_T; // フレーム飛ばし間隔
	float dist_direction; // センサから計算されるフレーム間の視線方向ベクトルの距離のしきい値

	// 各種フラグ
	bool f_comp = false; // 線形補完
	bool f_center = false; // センターサークル中心
	bool f_video = false; // ビデオ書き出し
	bool f_senser = false; // センサ情報の使用・不使用
	bool f_line = false; // 直線検出を利用
	bool f_undist = false; // レンズ歪み補正

	float fps = 20; // 書き出しビデオのfps
	string n_video; // 書き出しビデオファイル名
	string cam_data; // 映像センサファイル名
	string n_center; // センターサークル画像名
	string cam_param; // 内部パラメータのxmlファイル名
	string save_dir; // 各ファイルの保存先ディレクトリ

	try {
		// コマンドラインオプションの定義
		options_description opt("Usage");
		opt.add_options()("cam", value<std::string> (),
				"動画名やセンサファイル名が記述されたファイルの指定")("start,s",
				value<int> ()->default_value(0), "スタートフレームの指定")("end,e", value<
				int> ()->default_value(INT_MAX), "終了フレームの指定")("inter,i", value<
				int> ()->default_value(9), "取得フレームの間隔")("blur,b",
				value<int> ()->default_value(0), "ブラーによるフレームの破棄の閾値")("yaw",
				value<double> ()->default_value(0), "初期フレーム投影時のyaw")("hessian",
				value<int> ()->default_value(20), "SURFのhessianの値")("senser",
				value<double> ()->default_value(0.0), "センサー情報を使う際の視線方向のしきい値")(
				"line", value<bool> ()->default_value(false), "直線検出の利用")(
				"undist", value<bool> ()->default_value(false), "画像のレンズ歪み補正")(
				"outdir,o", value<string> ()->default_value("./"),
				"各種ファイルの出力先ディレクトリの指定")("cam_param", value<string> (),
				"内部パラメータ(.xml)ファイル名の指定")("help,h", "ヘルプの出力");

		// オプションのマップを作成
		variables_map vm;
		store(parse_command_line(argc, argv, opt), vm);
		notify(vm);

		// 必須オプションの確認
		if (vm.count("help")) {
			cout << "  [option]... \n" << opt << endl;
			return -1;
		}

		// 各種オプションの値を取得
		if (!vm.count("cam")) {
			cout << "cam_dataファイル名は必ず指定して下さい" << endl;
			return -1;
		}

		if (vm.count("cam_param")) { // 内部パラメータファイル名
			cam_param = vm["cam_param"].as<string> ();
		} else {
			cout << "内部パラメータファイル名を指定して下さい．" << endl;
			return -1;
		}

		if (vm.count("undist") && !vm.count("in_param")) {
			cout << "歪み補正をかけるには内部パラメータファイル名を指定して下さい．" << endl;
			return -1;
		}
		if (vm.count("senser")) {
			f_senser = true; // センサ情報の使用/不使用
			dist_direction = vm["senser"].as<float> ();
		} else {
			f_senser = true; // センサ情報の使用/不使用
			dist_direction = 0.0;
		}

		 cam_data = vm["cam"].as<string> (); // 映像 センサファイル名
		 skip = vm["start"].as<int> (); // 合成開始フレーム番号
		 end = vm["end"].as<int> (); // 合成終了フレーム番号
		 algorithm_type = vm["algo"].as<string> (); // 特徴点抽出記述アルゴリズム名
		 f_comp = vm["comp"].as<bool> (); // 補完の有効無効
		 FRAME_T = vm["inter"].as<int> (); // フレーム取得間隔
		 blur = vm["blur"].as<int> (); // 手ブレ閾値
		 fps = vm["fps"].as<int> (); // 書き出し動画のfps
		 hessian = vm["hessian"].as<int> (); // SURFのhessianパラメータ

		 f_undist = vm["undist"].as<bool> (); // レンズ歪み補正
		 save_dir = vm["outdir"].as<string> ();
		 f_line = vm["line"].as<bool> ();

	} catch (exception& e) {
		cerr << "error: " << e.what() << "\n";
		return -1;
	} catch (...) {
		cerr << "Exception of unknown type!\n";
		return -1;
	}

	feature = Feature2D::create(algorithm_type);
	if (algorithm_type.compare("SURF") == 0) {
		feature->set("extended", 1);
		feature->set("hessianThreshold", 10);
		feature->set("nOctaveLayers", 4);
		feature->set("nOctaves", 3);
		feature->set("upright", 0);
	}
	if (argc != 5) {
		cout << "Usage : " << argv[0] << "target_frame_image "
				<< "panorama_cam_data" << "panorama_background_image"
				<< "output_file_name" << "camera_param" << endl;
		return -1;
	}

	ifs_pano_cam.open(cam_data);
	if(!ifs_pano_cam.is_open()){
		cout << "cannot open pano_cam_data : " << cam_data << endl;
		return -1;
	}

	ifs_target_cam.open(cam_data);
	if (!ifs_target_cam.is_open()) {
		cerr << "cannnot open target_cam_data : " << argv[1] << endl;
		return -1;
	}

	// パノラマ背景の元動画，TIMEファイル，ORIファイルのパスを取得
	ifs_pano_cam >> str_pano_video;
	ifs_pano_cam >> str_pano_time;
	ifs_pano_cam >> str_pano_ori;

	// はめ込むフレームを含む動画，TIMEファイル，ORIファイルのパスを取得
	ifs_target_cam >> str_target_video;
	ifs_target_cam >> str_target_time;
	ifs_target_cam >> str_target_ori;

	// パノラマ背景画像のファイル名を取得
	str_pano = argv[3];

	// パノラマ背景画像の読み込み
	panorama = imread(str_pano);
	if (panorama.empty()) {
		cerr << "cannot open panorama image" << endl;
		return -1;
	}
	cout << "load background image" << endl;

	target_frame = imread(argv[1], CV_LOAD_IMAGE_COLOR); // 直接合成フレームを指定する場合argv[1]に画像pathを指定する

	// 合成対象のフレームのチャネルごとのヒストグラムを計算
	get_color_hist(target_frame, target_hist_channels);
	cout << "calc target_frame histgram" << endl;

	// パノラマ背景の元動画をオープン
	pano_cap.open(str_pano_video);
	if (!pano_cap.isOpened()) {
		cerr << "cannnot open panorama src movie" << endl;
		return -1;
	}

	//　パノラママスク画像の読み込み
	mask = imread("mask.jpg", CV_LOAD_IMAGE_GRAYSCALE);
	cout << "load background mask image" << endl;

	cout << "load target frame image" << endl;

	if (target_frame.empty()) {
		cerr << "cannnot load target frame" << endl;
	}

	pano_cap.set(CV_CAP_PROP_POS_FRAMES, s_pano);
	double pano_frame_time = pano_cap.get(CV_CAP_PROP_POS_MSEC);

	if (target_frame.empty()) {
		cerr << "cannnot open target frame" << endl;
		return -1;
	}

	// read camera pra
	Mat a_tmp;
	Mat dist;
	FileStorage cvfs_inparam(argv[5], CV_STORAGE_READ);
	FileNode node_inparam(cvfs_inparam.fs, NULL);

	read(node_inparam["intrinsic"], a_tmp);

	// create white_img
	white_img = Mat(target_frame.size(), CV_8U, Scalar::all(255));

	namedWindow("target", CV_WINDOW_NORMAL | CV_WINDOW_KEEPRATIO);
	imshow("target", target_frame);
	//imwrite("target.jpg", aim_frame);
	waitKey(30);
	namedWindow("panorama", CV_WINDOW_NORMAL | CV_WINDOW_KEEPRATIO);
	imshow("panorama", panorama);
	waitKey(30);
	Mat A1Matrix, A2Matrix;
	Mat yaw = Mat::eye(3, 3, CV_64FC1);
	Mat roll = cv::Mat::eye(3, 3, CV_64FC1);
	Mat pitch = cv::Mat::eye(3, 3, CV_64FC1);

	Mat hh = Mat::eye(3, 3, CV_64FC1);
	A1Matrix = Mat::eye(3, 3, CV_64FC1);

	cout << hh << endl;

	cout << "a" << endl;
	imshow("panorama", transform_image2);
	waitKey(30);

	// 合成したいフレームのORIからパノラマ平面へのホモグラフィを計算し重なり具合を計算
	/*string str_time(argv[6]);
	 string time_buf;
	 ifstream ifs_time(str_time.c_str());
	 double s_time;

	 if (!ifs_time.is_open()) {
	 cerr << "cannnot open TIME file : " << str_time << endl;
	 return -1;
	 }

	 ifs_time >> time_buf;
	 ifs_time >> time_buf;
	 ifs_time >> time_buf;
	 ifs_time >> s_time; // msec

	 s_time /= 1000.0;
	 cout << "s_time : " << s_time << endl;
	 SENSOR_DATA *sensor = (SENSOR_DATA *) malloc(sizeof(SENSOR_DATA) * 5000);

	 // 対象フレームのセンサデータを一括読み込み
	 LoadSensorData(argv[7], &sensor);

	 // 対象フレームの動画の頭からの時間frame_timeに撮影開始時刻s_timeを加算して，実時間に変換
	 aim_frame_time += s_time;
	 SENSOR_DATA pano_sd, aim_sd;
	 GetSensorDataForTime(aim_frame_time, &sensor, &aim_sd);

	 cout << "yaw : " << aim_sd.alpha << " pitch : " << aim_sd.beta
	 << " roll : " << aim_sd.gamma << endl;

	 // パノラマの時間を取得
	 ifs_time.close();
	 ifs_time.open(str_pano.c_str());
	 if (!ifs_time.is_open()) {
	 cerr << "cannnot open TIME file : " << str_time << endl;
	 return -1;
	 }

	 ifs_time >> time_buf;
	 ifs_time >> time_buf;
	 ifs_time >> time_buf;
	 ifs_time >> s_time; // msec

	 s_time /= 1000.0;
	 cout << "s_time : " << s_time << endl;

	 // センサデータを一括読み込み
	 LoadSensorData(argv[9], &sensor);

	 // 対象フレームの動画の頭からの時間frame_timeに撮影開始時刻s_timeを加算して，実時間に変換
	 pano_frame_time += s_time;

	 GetSensorDataForTime(aim_frame_time, &sensor, &aim_sd);

	 cout << "yaw : " << aim_sd.alpha << " pitch : " << aim_sd.beta
	 << " roll : " << aim_sd.gamma << endl;
	 */

	Mat result, r_result;
	// パノラマ画像と合成したいフレームの特徴点抽出と記述
	cout << "calc features" << endl;
	cvtColor(target_frame, gray_img1, CV_RGB2GRAY);
	cvtColor(panorama, gray_img2, CV_RGB2GRAY);
	erode(mask, mask2, cv::Mat(), cv::Point(-1, -1), 50);
	feature->operator ()(gray_img1, Mat(), objectKeypoints, objectDescriptors);
	feature->operator ()(gray_img2, mask2, imageKeypoints, imageDescriptors);

	//良い対応点の組みを求める
	good_matcher(objectDescriptors, imageDescriptors, &objectKeypoints,
			&imageKeypoints, &matches, &pt1, &pt2);
	cout << "selected good_matches : " << pt1.size() << endl;

	//mask = Mat(Size(PANO_W, PANO_H),CV_8UC3,Scalar::all(0));

	cout << "make drawmathces image" << endl;

	//ホモグラフィ行列を計算
	homography = findHomography(Mat(pt1), Mat(pt2), CV_RANSAC, 5.0);

	//合成したいフレームをパノラマ画像に乗るように投影
	warpPerspective(target_frame, transform_image, homography,
			Size(PANO_W, PANO_H), CV_INTER_LINEAR | CV_WARP_FILL_OUTLIERS);

	//投影場所のマスク生成
	warpPerspective(white_img, pano_black, homography, Size(PANO_W, PANO_H),
			CV_INTER_LINEAR | CV_WARP_FILL_OUTLIERS);
	namedWindow("panoblack", CV_WINDOW_NORMAL | CV_WINDOW_KEEPRATIO);
	imshow("panoblack", pano_black);
	waitKey(20);

	FileNode node(cvfs.fs, NULL); // Get Top Node

	Mat h_base;
	Mat tmp_base;
	stringstream ss;
	long n = 0;
	Mat aria1, aria2;
	Mat v1, v2;
	long max = LONG_MIN;
	long detect_frame_num;
	aria1 = Mat(Size(PANO_W, PANO_H), CV_8U, Scalar::all(0));
	aria2 = Mat(Size(PANO_W, PANO_H), CV_8U, Scalar::all(0));

	while (1) {
		//		frame_num = node["frame"];
		n++;
		ss << "homo_" << n;
		read(node[ss.str()], tmp_base);
		ss.clear();
		ss.str("");
		if (tmp_base.empty() && n < pano_cap.get(CV_CAP_PROP_FRAME_COUNT))
			continue;

		//ここに来たときtmp_baseに取得した行列，nにフレーム番号が格納されている
		// tmp_baseが空なら取り出し終わっているので下でブレイク処理に入る


		cout << n << " >= " << pano_cap.get(CV_CAP_PROP_FRAME_COUNT) << endl;
		if (n >= 180)
			break;

		warpPerspective(white_img, aria1, homography, Size(PANO_W, PANO_H),
				CV_INTER_LINEAR | CV_WARP_FILL_OUTLIERS);

		warpPerspective(white_img, aria2, tmp_base, Size(PANO_W, PANO_H),
				CV_INTER_LINEAR | CV_WARP_FILL_OUTLIERS);
		bitwise_and(aria1, aria2, aria1);
		namedWindow("mask", CV_WINDOW_NORMAL | CV_WINDOW_KEEPRATIO);
		imshow("mask", aria1);
		waitKey(0);
		threshold(aria1, aria2, 128, 1, THRESH_BINARY);

		long sum;
		sum = 0;
		for (unsigned long i = 0; i < aria2.rows * aria2.cols; i++)
			sum += aria2.data[i];
		if (sum > max) {
			max = sum;
			h_base = tmp_base.clone();
			detect_frame_num = n;
			cout << "detect near : " << detect_frame_num << endl;
		}

		double homo_norm, min_norm = DBL_MAX;
		homo_norm = norm(homography - tmp_base);
		if (homo_norm < min_norm) {
			min_norm = homo_norm;
			detect_frame_num = n;
			h_base = tmp_base.clone();
		}
		//tmp_base.release();

	}
	//free(sensor);

	cout << "detected near frame : " << detect_frame_num << endl;
	//cout << max << endl;

	//　重なりが大きいフレームを取り出す
	pano_cap.set(CV_CAP_PROP_POS_FRAMES, detect_frame_num);
	pano_cap >> near_frame;
	if (near_frame.empty()) {
		cerr << "cannot detect near frame" << endl;
		return -1;
	}

	ss << "homo_" << detect_frame_num;
	//ss << "homo_" << 3041;
	read(node[ss.str()], h_base);
	ss.clear();
	ss.str("");
	pano_cap.set(CV_CAP_PROP_POS_FRAMES, detect_frame_num);
	pano_cap >> near_frame;

	// 合成対象のフレームと最も近いフレームパノラマ背景のフレームのチャネルごとのヒストグラムを計算
	get_color_hist(target_frame, near_hist_channels);
	cout << "calc near_frame histgram" << endl;

	// 各チャネルに
	// 0.8-1.2間で0.1刻みでバイアスを変えながらヒストグラムの類似度を計算
	uchar lut[256];
	float est_gamma[3] = { 0.0, 0.0, 0.0 };
	double min_hist_distance[3] = { DBL_MAX, DBL_MAX, DBL_MAX };
	float gamma[3] = { 0.5, 0.5, 0.5 };
	vector<Mat> near_channels, target_channels;
	Mat gray_hist, tmp_channel, diff_img;
	split(near_frame, near_channels);
	split(target_frame, target_channels);

	// パノラマ背景の色合いを合成対象フレームに近づけるgammaを計算
	for (int i = 0; i < 3; i++) {

		// LUTの生成
		for (; gamma[i] < 2.0; gamma[i] += 0.001) {
			for (int j = 0; j < 256; j++)
				lut[j] = (j * gamma[i] <= 255) ? j * gamma[i] : 255;

			LUT(near_channels[i], Mat(Size(256, 1), CV_8U, lut), tmp_channel);
			get_gray_hist(tmp_channel, gray_hist);

			double hist_distance = compareHist(gray_hist,
					target_hist_channels[i], CV_COMP_CHISQR);
			//cout << "dist : " << hist_distance << endl;
			if (hist_distance < min_hist_distance[i]) {
				est_gamma[i] = gamma[i];
				min_hist_distance[i] = hist_distance;
			}
		}
	}

	cout << " <est_gamma> " << endl;
	cout << est_gamma[0] << " : " << est_gamma[1] << " : " << est_gamma[2]
			<< endl;

	for (int i = 0; i < 3; i++) {
		for (int j = 0; j < 256; j++)
			lut[j] = (j * est_gamma[i] <= 255) ? j * est_gamma[i] : 255;
		LUT(near_channels[i], Mat(Size(256, 1), CV_8U, lut), result);
		near_channels[i] = result.clone();
	}

	Mat fix_near_image;
	merge(near_channels, fix_near_image);
	LUT(transform_image2, Mat(Size(256, 1), CV_8U, lut), result);
	//transform_image2 = result.clone();
	imwrite("fix_test.jpg", transform_image2);

	Mat hough_src, hough_dst; // ハフ変換の入力と，検出した線の出力先
	cvtColor(near_frame, gray_img2, CV_RGB2GRAY);
	Canny(gray_img2, hough_src, 30, 50, 3);

	imshow("detected lines", hough_src);
	//			waitKey(0);

	hough_dst = Mat(hough_src.size(), CV_8U, Scalar::all(0));

	vector<Vec4i> lines;
	HoughLinesP(hough_src, lines, 1, CV_PI / 180, 80, 100, 200);

	for (size_t i = 0; i < lines.size(); i++) {
		Vec4i l = lines[i];
		line(hough_dst, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(255, 255,
				255), 3, CV_AA);
	}

	// 検出した直線を膨張させて，マスク画像を作成
	Mat t = hough_dst;
	dilate(t, hough_dst, cv::Mat(), cv::Point(-1, -1), 5);
	hough_dst = Mat(hough_src.size(), CV_8U, Scalar::all(255));
	for (int i = 0; i < hough_dst.cols; i++)
		for (int j = 0; j < hough_dst.rows; j++) {
			if (i > hough_dst.cols / 2.5 || i < hough_dst.cols / 8)
				hough_dst.at<unsigned char> (j, i) = 0;
			if (j < hough_dst.rows / 8 || j > hough_dst.rows * 3.0 / 4.0)
				hough_dst.at<unsigned char> (j, i) = 0;
		}
	cvtColor(near_frame, gray_img2, CV_RGB2GRAY);
	feature->operator ()(gray_img2, Mat(), imageKeypoints, imageDescriptors);

	good_matcher(objectDescriptors, imageDescriptors, &objectKeypoints,
			&imageKeypoints, &matches, &pt1, &pt2);
	drawMatches(target_frame, objectKeypoints, near_frame, imageKeypoints,
			matches, result);

	//resize(result, r_result, Size(), 0.5, 0.5, INTER_LANCZOS4);

	cout << "show matches" << endl;
	namedWindow("matches", CV_WINDOW_NORMAL | CV_WINDOW_KEEPRATIO);
	imshow("matches", result);
	waitKey(20);
	homography = findHomography(Mat(pt1), Mat(pt2), CV_RANSAC, 5.0);
	warpPerspective(white_img, pano_black, h_base * homography, Size(PANO_W,
			PANO_H), CV_INTER_LINEAR | CV_WARP_FILL_OUTLIERS);
	warpPerspective(target_frame, transform_image, h_base * homography, Size(
			PANO_W, PANO_H), CV_INTER_LINEAR | CV_WARP_FILL_OUTLIERS);
	FileStorage fs("homography.xml", cv::FileStorage::WRITE);
	write(fs, "homography", h_base * homography);
	namedWindow("transform_image", CV_WINDOW_NORMAL | CV_WINDOW_KEEPRATIO);
	imshow("transform_image", transform_image);
	waitKey(0);

	/*
	 //投影先での対応点を再計算
	 bitwise_and(mask, pano_black, mask);
	 erode(mask, mask2, cv::Mat(), cv::Point(-1, -1), 30);
	 feature->operator ()(gray_img2, mask2, objectKeypoints, objectDescriptors);
	 good_matcher(imageDescriptors, objectDescriptors, &imageKeypoints,
	 &objectKeypoints, &matches, &pt1, &pt2);
	 homography = findHomography(Mat(pt1), Mat(pt2), CV_RANSAC, 5.0);
	 warpPerspective(white_img, pano_black, homography, Size(PANO_W, PANO_H),
	 CV_INTER_LINEAR | CV_WARP_FILL_OUTLIERS);
	 drawMatches(aim_frame, imageKeypoints, panorama, objectKeypoints, matches,
	 result);
	 resize(result, r_result, Size(), 0.5, 0.5, INTER_LANCZOS4);

	 cout << "show matches" << endl;
	 namedWindow("matches", CV_WINDOW_NORMAL | CV_WINDOW_KEEPRATIO);
	 imshow("matches", result);
	 waitKey(30);
	 */

	bitwise_and(mask, pano_black, mask);
	imshow("panoblack", transform_image2);
	waitKey(30);
	make_pano(transform_image, transform_image2, ~mask, pano_black);

	namedWindow("result", CV_WINDOW_NORMAL | CV_WINDOW_KEEPRATIO);
	imshow("result", transform_image2);
	waitKey(30);
	imwrite("result.jpg", transform_image2);

	return 0;
}
