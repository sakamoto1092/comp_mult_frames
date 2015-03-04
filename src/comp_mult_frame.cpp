/*
 *  単一のカメラで撮影し，生成したパノラマ背景画像に
 *  同一カメラで同じ位置から撮影した指定の１枚のフレーム画像をはめ込むプログラム
 *  （背景とはめ込むフレームは同一の内部パラメータ）
 *
 *  はめ込むフレーム画像は，動画中のフレームまたは画像ファイルで指定可能
 *  前提として，
 *  ・カメラの内部パラメータ及び歪み係数を記述したxmlファイル
 *  ・パノラマ背景生成時のホモグラフィ行列を記述したxmlファイル
 *  ・パノラマ背景を生成した際に得られるマスク画像
 *  が必要
 */
#include <opencv2/opencv.hpp>
#include <opencv2/stitching/stitcher.hpp>
#include <opencv2/nonfree/nonfree.hpp>
#include <opencv2/stitching/detail/blenders.hpp>

#include<AKAZE.h>

#include <boost/program_options.hpp>

#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>

#include <sstream>
#include <string>
#include <iostream>
#include <iomanip>
#include <vector>
#include <fstream>
#include <map>

#include "3dms-func.h"

// パノラマ画像の大きさ
#define PANO_W 6000
#define PANO_H 3000

// 合成フレームのFPS
#define TARGET_VIDEO_FPS 30
#define INPUT_FRAME_FPS 30

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

	VideoCapture target_cap; // target frame movie
	VideoCapture pano_cap; // background frame movie
	FileStorage cvfs("log.xml", CV_STORAGE_READ); // パノラマ背景生成時のホモグラフィ行列のログファイル
	FileNode node(cvfs.fs, NULL); // Get Top Node

	Mat panorama, target_frame, near_frame; // パノラマ画像，合成対象フレーム画像，近傍背景画像
	Mat homography; // ホモグラフィ行列
	Mat senserbased_homographty;
	Mat calced_A1, calced_A2;
	Mat dist_src;
	ifstream ifs_target_cam; // target_camファイルストリーム
	ifstream ifs_pano_cam; // pano_cam　ファイルストリーム
	long s_pano;

	//	string str_pano, str_frame, str_video;
	string str_pano_video, str_pano_time, str_pano_ori; // pano_cam内の各種ファイル名
	string str_target_frame,str_target_video, str_target_time, str_target_ori; // target_cam内のの各種ファイル名
	string str_pano; // パノラマ背景画像のファイル名
	stringstream ss; // 汎用文字列ストリーム
	Mat transform_image; // 画像単体でのパノラマ背景への変換結果
	Mat transform_image2 = Mat(Size(PANO_W, PANO_H), CV_8UC3); // 合成中のパノラマ画像

	vector<Mat> target_hist_channels; // 合成対象のフレーム画像のチャネルごとのヒストグラム
	vector<Mat> near_hist_channels; // 合成対照フレーム近傍の背景画像のチャネルごとのヒストグラム

	Mat mask = Mat(Size(PANO_W, PANO_H), CV_8U, Scalar::all(0)); // パノラマ画像のマスク
	Mat pano_black = Mat(Size(PANO_W, PANO_H), CV_8U, Scalar::all(0)); // パノラマ画像と同じサイズの黒画像
	Mat white_img; // = Mat(Size(640, 360), CV_8U, Scalar::all(255)); // フレームと同じサイズの白画像
	Mat gray_img1, gray_img2; // 特徴点に利用するグレースケール画像
	Mat mask2; // 膨張縮小処理後のマスク画像

	//　対応点座標に関する変数
	vector<Point2f> pt1, pt2; // 特徴点の対の画像上座標
	std::vector<cv::DMatch> matches; // 対応点の対の格納先

	// 特徴点の集合と特徴量 (object画像をimage画像上に射影するイメージ)
	std::vector<KeyPoint> objectKeypoints, imageKeypoints;
	Mat objectDescriptors, imageDescriptors;

	//　特徴点に関する変数

	string algorithm_type("SIFT"); // 特徴点抽出および特徴量記述のアルゴリズム名
	Ptr<Feature2D> feature; // 汎用特徴点抽出および特徴量記述クラスへのポインタ
	int hessian; // SURF特徴検出の際に用いるしきい値（小さいほど特徴点が検出されやすい）

	//PyramidAdaptedFeatureDetector g_feature_detector(new SiftFeatureDetector,5);
	//GridAdaptedFeatureDetector g_feature_detector(new SiftFeatureDetector(),1000000, 2, 2);

	//if (g_feature_detector.empty()) {
	//	cout << "faild init g_feature_detector." << endl;
	//}

	long skip; 				// 合成開始フレーム番号
	long end; 				// 合成終了フレーム番号
	//long frame_num; 		// 現在のフレーム位置
	int blur; // ブレのしきい値
	//long FRAME_MAX; 		// 動画の最大フレーム数
	long FRAME_T; // フレーム飛ばし間隔
	float dist_direction; // センサから計算されるフレーム間の視線方向ベクトルの距離のしきい値
	long target_frame_num; // 動画のフレームをはめ込む場合のフレーム指定
	double target_frame_time; // 動画のフレームの先頭からの時間
	double pano_frame_time;

	// 各種フラグ
	//bool f_comp = false;  			// 線形補完
	//bool f_center = false; 		// センターサークル中心
	//bool f_video = false; 			// ビデオ書き出し
	bool f_senser = false; // センサ情報の使用・不使用
	bool f_line = false; // 直線検出を利用
	bool f_undist = false; // レンズ歪み補正
	bool f_target_video = false; // はめ込み対象が動画ファイル（or １枚のフレーム画像）
	int f_adj_color = 0; // 色調補正の有無(0:補正なし 1:背景を補正 2:フレームを補正 3:見本画像に両方合わせる)

	//float fps = 20; 				// 書き出しビデオのfps
	//string n_video; 				// 書き出しビデオファイル名
	string pano_cam_path; 			// パノラマ背景cam_dataセンサファイル名
	string target_cam_path; 			// はめ込み対象cam_dataファイル名
	//string n_center; 				// センターサークル画像名
	string pano_cam_param_path; 	// パノラマ背景合成に使った映像の内部パラメータファイル名
	string target_cam_param_path;	// はめ込む映像の内部パラメータファイル名
	string save_path; 				// 各ファイルの保存先ディレクトリ名
	string example_path; 		// 色調補正の目標画像（ここで指定された画像のヒストグラムにあわせて両方に補正をかける）
	string result_name; 				// 処理結果画像ファイルの名前

	time_t timer;        // 日付取得用
	struct tm *local;    // 日付取得用

	struct timeval t0, t1; // 処理時間計算用変数 t0:start t1:end

	timer = time(NULL);			// 現在時間の取得
	local = localtime(&timer);  // 地方時間に変換
	gettimeofday(&t0, NULL); 	// 開始時刻の取得

	//TickMeter in OpenCV
	TickMeter tmeter;


	Mat h_base;
	Mat tmp_base;
	long n = 0;
	Mat aria1, aria2;
	Mat v1, v2;
	long max = LONG_MIN;
	long detect_frame_num;
	double min = DBL_MAX;

	Mat rollMatrix = cv::Mat::eye(3, 3, CV_64FC1);
	Mat pitchMatrix = cv::Mat::eye(3, 3, CV_64FC1);
	Mat yawMatrix = cv::Mat::eye(3, 3, CV_64FC1);

	Mat a_tmp;
	Mat dist;
	list<pair<int, SENSOR_DATA> > near_senser_list; // はめ込み対象に視線方向が近いフレーム番号とそのセンサ情報を格納する

	try {
		// コマンドラインオプションの定義
		options_description opt("Usage");
		opt.add_options()("panorama", value<std::string>(), "パノラマ背景画像ファイルの指定")
				("pano_cam", value<std::string>(), "パノラマ背景のcam_dataファイルの指定")
				("target_cam", value<std::string>(), "合成対象のcam_dataファイルの指定")
				("target_num", value<long>()->default_value(0),"合成対象の動画のフレーム番号の指定")
				("target_frame", value<std::string>(),"合成対象のcam_dataファイルの指定(静止画用)")
				("start,s", value<long>()->default_value(0),"スタートフレームの指定")
				("end,e", value<long>()->default_value(0),"終了フレームの指定")
				("inter,i", value<long>()->default_value(9),"取得フレームの間隔")
				("blur,b", value<int>()->default_value(0),"ブラーによるフレームの破棄の閾値")
				("yaw", value<double>()->default_value(0),"初期フレーム投影時のyaw")
				("hessian", value<int>()->default_value(20),"SURFのhessianの値")
				("senser", value<double>()->default_value(0.0),"センサー情報を使う際の視線方向のしきい値")
				("line",value<bool>()->default_value(false), "直線検出の利用")
				("undist",value<bool>()->default_value(false), "画像のレンズ歪み補正")
				("outdir",value<string>()->default_value("./"), "各種ファイルの出力先ディレクトリの指定")
				("pano_cam_param", value<string>(),"パノラマ背景合成に使った映像の内部パラメータファイル(.xml)の指定")
				("target_cam_param",value<string>(), "はめ込む映像の内部パラメータのファイル(.xml)の指定")
				("adj_color",value<string>(), "パノラマ背景またははめ込み対象フレーム画像の色味を合わせる")
				("out,o",value<string>()->default_value("target.jpg"), "処理結果画像ファイルの名前")
				("help,h", "ヘルプの出力");

		// オプションのマップを作成
		variables_map vm;
		store(parse_command_line(argc, argv, opt), vm);
		notify(vm);

		// 必須オプションの確認
		// ヘルプの表示
		if (vm.count("help")) {
			cerr << "  [option]... \n" << opt << endl;
			return -1;
		}
		// 各種オプションの値を取得の確認
		if (!vm.count("pano_cam")) {
			cerr << "cam_dataファイル名は必ず指定して下さい" << endl;
			return -1;
		}
		//パノラマ背景映像の 内部パラメータファイル名の入力の確認
		if (vm.count("pano_cam_param")) {
			pano_cam_param_path = vm["pano_cam_param"].as<string>();
		} else {
			cerr << "パノラマ背景合成時の内部パラメータファイル名を指定して下さい．" << endl;
			return -1;
		}

		// はめ込む映像の内部パラメータファイル名の入力の確認
		if (vm.count("target_cam_param")) {
			target_cam_param_path = vm["target_cam_param"].as<string>();
		} else {
			cerr << "はめ込む映像の内部パラメータファイル名を指定して下さい．" << endl;
			return -1;
		}

		if (!vm.count("target_cam") && !vm.count("target_frame")) {
			cerr << "はめ込み対象を指定して下さい．" << endl;
			cerr << "select target_cam or target_frame" << endl;
			return -1;
		}

		// レンズ歪みを修正するための，歪み系数を含む内部パラメータファイルの確認
		if (vm.count("undist") && !vm.count("target_cam_param")) {
			cerr << "歪み補正をかけるには内部パラメータファイル名を指定して下さい．" << endl;
			return -1;
		}

		if (!vm.count("panorama")) {
			cout << "パノラマ背景画像を指定して下さい．" << endl;
			return -1;
		}

		// センサ情報から視線方向を推定し合成に利用するかの確認
		if (vm.count("senser")) {
			f_senser = true; // センサ情報の使用/不使用
			dist_direction = vm["senser"].as<double>();
		} else {
			f_senser = false; // センサ情報の使用/不使用
			dist_direction = 0.0;
		}

		// はめ込み対象にcam_dataファイルを指定する場合(同時にフレーム番号も設定して指定)
		if (vm.count("target_cam")) {
			f_target_video = true;
			target_cam_path = vm["target_cam"].as<string>();
			if (!vm.count("target_num")) {
				cerr<< "合成対象をcam_dataで指定する場合は，合成対象フレームの番号(--target_num)を指定して下さい．"<< endl;
				cerr << "合成対象フレーム番号が設定されていないので先頭フレームをはめ込みます．" << endl;
			}
			target_frame_num = vm["target_num"].as<long>();
		}

		// はめ込み対象に画像ファイルを指定する場合(cam_dataはセンサと撮影時刻の情報として使われる)
		if (vm.count("target_frame")) {
			f_target_video = false;
			target_cam_path = vm["target_frame"].as<string>();

			if (!vm.count("target_num")) {
				cerr<< "合成対象をcam_frameで指定する場合は，合成対象フレームの番号(--target_num)を指定して下さい．"<< endl;
				cerr << "合成対象フレーム番号が設定されていないので先頭フレームをはめ込みます．" << endl;
			}
			target_frame_num = vm["target_num"].as<long>();
		}

		// 色味を補正する場合
		if (vm.count("adj_color")) {
			example_path = vm["adj_color"].as<string>();
			if (example_path.compare("1") == 0)
				f_adj_color = 1; // 背景を補正
			else if (example_path.compare("2") == 0)
				f_adj_color = 2; // フレームを補正
			else
				f_adj_color = 3; // 見本に両方合わせる
		}

		pano_cam_path = vm["pano_cam"].as<string>(); // 映像 センサファイル名(default : cam_data.txt)
		skip = vm["start"].as<long>(); 				// 合成開始フレーム番号 (default : 0)
		end = vm["end"].as<long>();			// 合成終了フレーム番号 (default : INT_MAX)
		// algorithm_type = vm["algo"].as<string> ();	// 特徴点抽出記述アルゴリズム名 (default : SURF)
		//f_comp = vm["comp"].as<bool> (); 			// 補完の有効無効

		FRAME_T = vm["inter"].as<long>(); 			// フレーム取得間隔
		//blur = vm["blur"].as<int> (); 					// 手ブレ閾値
		//fps = vm["fps"].as<int> (); 					// 書き出し動画のfps
		hessian = vm["hessian"].as<int>(); // SURFのhessianパラメータ
		f_undist = vm["undist"].as<bool>(); // レンズ歪み補正のフラグ (default : OFF)
		save_path = vm["outdir"].as<string>(); // 各種ファイルの出力先の指定 (default : "./")
		f_line = vm["line"].as<bool>(); // 確率的ハフ変換による特徴点のマスク
		str_pano = vm["panorama"].as<string>(); // パノラマ背景画像のファイル名
		result_name = vm["out"].as<string>(); // 処理結果画像のファイル名を取得
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
		feature->set("hessianThreshold", hessian);
		feature->set("nOctaveLayers", 4);
		feature->set("nOctaves", 3);
		feature->set("upright", 0);
	} else if (algorithm_type.compare("SIFT") == 0) {
		//feature->set("nfeatures", 0);
		feature->set("nOctaveLayers", 10);
		feature->set("contrastThreshold", 0.02);
		feature->set("edgeThreshold", 20);
		feature->set("sigma", 2.0);
	}

	// パノラマ背景の生成に利用したcam_dataファイルの読み込み
	ifs_pano_cam.open(pano_cam_path.c_str());
	if (!ifs_pano_cam.is_open()) {
		cerr << "cannot open pano_cam_data : " << pano_cam_path << endl;
		return -1;
	}

	// はめ込み対象フレームを動画から取得する場合
	// 対象動画のcam_dataファイルを読み込む
	ifs_target_cam.open(target_cam_path.c_str());
	if (!ifs_target_cam.is_open()) {
		cerr << "cannnot open target_cam_data : " << target_cam_path
				<< endl;
		return -1;
	}


	// パノラマ背景の元動画，TIMEファイル，ORIファイルのパスを取得
	ifs_pano_cam >> str_pano_video;
	ifs_pano_cam >> str_pano_time;
	ifs_pano_cam >> str_pano_ori;


	// はめ込むフレームを含む動画ファイル，TIMEファイル，ORIファイルのパス，合成対象フレーム番号を取得
	if (f_target_video) {
		ifs_target_cam >> str_target_video;
		ifs_target_cam >> str_target_time;
		ifs_target_cam >> str_target_ori;
	}else{
		ifs_target_cam >> str_target_frame;
		ifs_target_cam >> str_target_time;
		ifs_target_cam >> str_target_ori;
	}

	// 内部パラメータファイル読み込み用変数
	FileStorage cvfs_inparam(pano_cam_param_path, CV_STORAGE_READ);
	FileNode node_inparam(cvfs_inparam.fs, NULL);

	/*
	 // パノラマ画像の合成に使われたセンサ情報の読み込み
	 FileNodeIterator log_it = node.begin();
	 FileNodeIterator log_end = node.end();
	 size_t max_node = node.size();
	 size_t read_count = 0;

	 vector<int> v_pano_frame_nums; // パノラマ背景の合成に使われたフレーム画像の番号
	 while (read_count != max_node) {
	 int i = 0;
	 ss << "homo_" << i;
	 read(node[ss.str()], tmp_base);
	 ss.clear();
	 ss.str("");

	 // home_iというホモグラフィ行列が取得できたならtmp_baseはempty()はfalseを返す
	 if (tmp_base.empty()) {
	 ++i;
	 continue;
	 }
	 ++read_count;

	 // パノラマに使われたフレーム番号を取得
	 v_pano_frame_nums.push_back(i);

	 i++;
	 }
	 */
	// パノラマ背景画像の読み込み
	//cout << "load background image" << endl;
	panorama = imread(str_pano);
	if (panorama.empty()) {
		cerr << "cannot open panorama image" << endl;
		return -1;
	}
	//cout << "done" << endl;

	// パノラマ背景の元動画をオープン
	//cout << "open panorama video" << endl;
	pano_cap.open(str_pano_video);
	if (!pano_cap.isOpened()) {
		cerr << "cannnot open panorama src movie" << endl;
		return -1;
	}
	//cout << "done" << endl;

	//　パノラママスク画像の読み込み
	//cout << "load background mask image" << endl;
	mask = imread("mask.jpg", CV_LOAD_IMAGE_GRAYSCALE);
	if (mask.empty()) {
		cerr << "cannot open mask image" << endl;
		return -1;
	}
	//cout << "done" << endl;

	// 合成対象フレーム画像の読み込み
	//cout << "load target frame image..." << endl;
	if (!f_target_video) {
		// 画像ファイルを指定
		ss.str("");
		ss.clear();
		ss << str_target_frame << "frame-" << setfill('0') << setw(5) << target_frame_num << ".jpg";
		target_frame = imread(ss.str(), CV_LOAD_IMAGE_COLOR);
		//resize(target_frame,target_frame,Size(target_frame.cols/2.0,target_frame.rows/2.0));
		target_frame_time = (float)INPUT_FRAME_FPS*(float)target_frame_num / 1000.0f;
		if (target_frame.empty()) {
			cerr << "cannnot load target frame : " << ss.str() << endl;
			return -1;
		}
		//cout << "loaded image file." << endl;
		//cout << "target frame : " << target_frame_num << "[frame]" << "time : "
		//		<< target_frame_time << "[sec]" << endl;
	} else {
		// cam_dataファイルから動画を開いてフレームを取得
		target_cap.open(str_target_video);
		//target_cap.set(CV_CAP_PROP_POS_FRAMES, target_frame_num);
		//	target_cap >> target_frame;
		for (int i = 0; i < target_frame_num - 1; i++)
			target_cap.grab();
		target_cap >> target_frame;

		target_frame_time = target_cap.get(CV_CAP_PROP_POS_MSEC) / 1000.0;
		if (target_frame.empty()) {
			cerr << "cannnot load target frame from video : " << endl;
			cerr << str_target_video << " " << target_cam_path << "[frame]"
					<< endl;
			return -1;
		}
		//cout << "loaded from video file" << endl;
		//cout << "target frame : " << target_frame_num << "[frame]" << "time : "
		//		<< target_frame_time << "[sec]" << endl;
	}
	//resize(target_frame,target_frame,Size(target_frame.cols/2.0,target_frame.rows/2.0));
	//cout << "done" << endl;
	//target_frame = imread("2.jpg", CV_LOAD_IMAGE_COLOR);
	// 合成対象のフレームのチャネルごとのヒストグラムを計算
	if (f_adj_color != 0) {
		// 色見補正のオプションが指定されていたら
		cout << "calc target_frame histgram" << endl;
		get_color_hist(target_frame, target_hist_channels);
		cout << "done" << endl;
	}
	/*
	 pano_cap.set(CV_CAP_PROP_POS_FRAMES, s_pano);
	 double pano_frame_time = pano_cap.get(CV_CAP_PROP_POS_MSEC);
	 */

	// 各種ウインドウの初期化
	//namedWindow("target", 			CV_WINDOW_NORMAL | CV_WINDOW_KEEPRATIO);
	//namedWindow("panorama",		 	CV_WINDOW_NORMAL | CV_WINDOW_KEEPRATIO);
	//namedWindow("panoblack",			CV_WINDOW_NORMAL | CV_WINDOW_KEEPRATIO);
	//namedWindow("base frame", 				CV_WINDOW_NORMAL | CV_WINDOW_KEEPRATIO);
	//namedWindow("matches", 			CV_WINDOW_NORMAL | CV_WINDOW_KEEPRATIO);
	//namedWindow("transform_image", 	CV_WINDOW_NORMAL | CV_WINDOW_KEEPRATIO);
	//namedWindow("result",			CV_WINDOW_NORMAL | CV_WINDOW_KEEPRATIO);
	// read camera param
	// create white_img
	white_img = Mat(target_frame.size(), CV_8U, Scalar::all(255));

	//imshow("target", target_frame);
	//waitKey(30);
	if (f_target_video) {
		ss.str("");
		ss.clear();
		ss << save_path << "target-" << setfill('0') << setw(5) << right
				<< target_frame_num << ".jpg";
		imwrite(ss.str(), target_frame);
		cout << "save target frame from video to : " << ss.str() << endl;
	}

	Mat A1Matrix, A2Matrix;         // A1 : panorama_param (パノラマの元画像)
									// A2 : object_param (はめ込む対象フレーム)
	Mat yaw = Mat::eye(3, 3, CV_64FC1);
	Mat roll = Mat::eye(3, 3, CV_64FC1);
	Mat pitch = Mat::eye(3, 3, CV_64FC1);

	Mat sensing_rotaion1 = Mat::eye(3, 3, CV_64FC1); // センサーによるROIから計算した回転行列(panorama_rotaion)
	Mat sensing_rotaion2 = Mat::eye(3, 3, CV_64FC1); // センサーによるROIから計算した回転行列(object_rotaion)

	Mat hh = Mat::eye(3, 3, CV_64FC1);
	A1Matrix = Mat::eye(3, 3, CV_64FC1);

	// 各内部パラメータの取得
	read(node_inparam["intrinsic"], a_tmp);
	read(node_inparam["distortion"], dist);
	A1Matrix = a_tmp.clone();
	//A1Matrix = a_tmp.clone();

	cvfs_inparam.open(target_cam_param_path, CV_STORAGE_READ);
	node_inparam = FileNode(cvfs_inparam.fs, NULL);

	read(node_inparam["intrinsic"], a_tmp);
	read(node_inparam["distortion"], dist);
	A2Matrix = a_tmp.clone();

	//cout << "load test cam_param" << endl << endl;
	//cout << A1Matrix << endl;
	//cout << A2Matrix << endl;

	//cout << hh << endl;
	transform_image2 = panorama.clone();
	//imshow("panorama", transform_image2);
	//waitKey(30);

	// 合成したいフレームのORIからパノラマ平面へのホモグラフィを計算し重なり具合を計算

	string time_buf;
	ifstream ifs_time(str_target_time.c_str());
	double s_time;

	if (!ifs_time.is_open()) {
		cerr << "cannnot open TIME file : " << str_target_time << endl;
		return -1;
	}

	ifs_time >> time_buf;
	ifs_time >> time_buf;
	ifs_time >> time_buf;
	ifs_time >> s_time; // msec

	s_time /= 1000.0;
	//cout << "s_time : " << s_time << endl;
	//printf("%f\n", s_time);
	SENSOR_DATA *sensor = (SENSOR_DATA *) malloc(	sizeof(SENSOR_DATA) * MAXDATA_3DMS);

	//cout << "load target sensor data..." << endl;
	// 対象フレームのセンサデータを一括読み込み
	LoadSensorData(str_target_ori.c_str(), &sensor);
	//cout << "done" << endl;

	// 対象フレームの動画の頭からの時間frame_timeに撮影開始時刻s_timeを加算して，実時間に変換
	target_frame_time += s_time;
	SENSOR_DATA pano_sd, aim_sd, near_sd;
	GetSensorDataForTime(target_frame_time, &sensor, &aim_sd);

	cout << "yaw : " << aim_sd.alpha << " pitch : " << aim_sd.beta << " roll : "<< aim_sd.gamma << endl;



	//cout << " time diff : " << aim_sd.TT - s_time << endl;
	// ORIからの回転行列を計算
	SetYawRotationMatrix(&yawMatrix, aim_sd.alpha);
	SetPitchRotationMatrix(&pitchMatrix, aim_sd.beta);
	SetRollRotationMatrix(&rollMatrix, aim_sd.gamma);

	sensing_rotaion2 = rollMatrix * pitchMatrix * yawMatrix;
	;
	//cout << "target_rotation \n" << sensing_rotaion2 << endl;

	// パノラマの時間を取得
	ifs_time.close();
	ifs_time.open(str_pano_time.c_str());
	if (!ifs_time.is_open()) {
		cerr << "cannnot open TIME file : " << str_pano_time << endl;
		return -1;
	}

	ifs_time >> time_buf;
	ifs_time >> time_buf;
	ifs_time >> time_buf;
	ifs_time >> s_time; // msec

	s_time /= 1000.0;   //convert to sec
	//cout << "s_time : " << s_time << endl;
	//printf("%f\n", s_time);
	// センサデータを一括読み込み
	LoadSensorData(str_pano_ori.c_str(), &sensor);

	/*
	 // 対象フレームの動画の頭からの時間frame_timeに撮影開始時刻s_timeを加算して，実時間に変換
	 pano_frame_time += s_time;

	 GetSensorDataForTime(aim_frame_time, &sensor, &aim_sd);

	 cout << "yaw : " << aim_sd.alpha << " pitch : " << aim_sd.beta
	 << " roll : " << aim_sd.gamma << endl;
	 */

	Mat result, r_result;
	// パノラマ画像と合成したいフレームの特徴点抽出と記述
	//cout << "calc features" << endl;
	dist_src = target_frame.clone();
	//undistort(dist_src,target_frame,A1Matrix,dist);  //悪くなるんでやらない

	Mat half_target_frame; // 高速化のために1/2解像度で特徴点検出

	//resize(target_frame,target_frame,Size(target_frame.cols/2.0,target_frame.rows/2.0));

	cvtColor(target_frame, gray_img1, CV_RGB2GRAY);
	//cvtColor(half_target_frame, gray_img1, CV_RGB2GRAY);
	//cvtColor(panorama, gray_img2, CV_RGB2GRAY);
	//erode(mask, mask2, cv::Mat(), cv::Point(-1, -1), 50);

	//feature.detect(gray_img1, Mat(), objectKeypoints, objectDescriptors);
	//g_feature_detector.detect(gray_img1,objectKeypoints);
/*
	AKAZEOptions akaze_op;
	akaze_op.img_height = target_frame.rows;
	akaze_op.img_width = target_frame.cols;
	akaze_op.dthreshold = 0.0001f;
	AKAZE akaze(akaze_op);
	*/
	/*
	 {
	 Mat img32;
	 gray_img1.convertTo(img32,CV_32F,1.0/255.0);
	 akaze.Create_Nonlinear_Scale_Space(img32);
	 akaze.Feature_Detection(objectKeypoints);
	 akaze.Compute_Descriptors(objectKeypoints,objectDescriptors);
	 }
	 */
	feature->operator ()(gray_img1, Mat(), objectKeypoints, objectDescriptors,false);

	SetYawRotationMatrix(&yawMatrix, aim_sd.alpha);
	SetPitchRotationMatrix(&pitchMatrix, aim_sd.beta);
	SetRollRotationMatrix(&rollMatrix, aim_sd.gamma);
	Mat vec1(3, 1, CV_64F), vec2(3, 1, CV_64F);
	vec1 = yawMatrix * pitchMatrix * rollMatrix* (cv::Mat_<double>(3, 1) << 1, 0, 0);



	while (1) {
		n++;
		ss << "homo_" << n;
		read(node[ss.str()], tmp_base);
		ss.clear();
		ss.str("");
		if (tmp_base.empty() && n < pano_cap.get(CV_CAP_PROP_FRAME_COUNT)) {
			//cout << "------------------------" << endl;
			continue;
		}
		//ここに来たときtmp_baseに取得した行列，nにフレーム番号が格納されている
		if (n >= pano_cap.get(CV_CAP_PROP_FRAME_COUNT)) //最後のフレームのホモグラフィー行列が異常なので対象から省く必要があるものがある
			break;

		pano_cap.set(CV_CAP_PROP_POS_FRAMES, n);
		pano_frame_time = pano_cap.get(CV_CAP_PROP_POS_MSEC) / 1000.0;
		pano_frame_time += s_time; // s_timeにはパノラマ背景の元動画の撮影開始時間が入っている[sec]

		GetSensorDataForTime(pano_frame_time, &sensor, &pano_sd);

		//cout << " pano_sd yaw : " << pano_sd.alpha << " pitch : "
		//		<< pano_sd.beta << " roll : " << pano_sd.gamma << endl;
		//cout << pano_frame_time - s_time << "[sec]" << endl;

		SetYawRotationMatrix(&yawMatrix, pano_sd.alpha);
		SetPitchRotationMatrix(&pitchMatrix, pano_sd.beta);
		SetRollRotationMatrix(&rollMatrix, pano_sd.gamma);

		vec2 = yawMatrix * pitchMatrix * rollMatrix	* (cv::Mat_<double>(3, 1) << 1, 0, 0);

		double distanse = sqrt(pow(vec1.at<double>(0, 0) - vec2.at<double>(0, 0), 2)
						+ pow(vec1.at<double>(0, 1) - vec2.at<double>(0, 1), 2)
						+ pow(vec1.at<double>(0, 2) - vec2.at<double>(0, 2),2));
		//cout << "dist : " << distanse << endl;
		//distanse = abs(pano_sd.alpha - aim_sd.alpha);
		// 近いものがあったらnear_sdを更新
		if (distanse < min || true) {
			min = distanse;
			//near_sd = pano_sds[i];
			//cout << "detect near frame : " << n << endl;
			//f_detect_near = true;
			//near_frame = vec_n_pano_frames[i];
			//near_homography = pano_monographys[i].clone();
			detect_frame_num = n;
			near_senser_list.push_front(make_pair((int) n, pano_sd)); // 良いものリストを作る(パノラマ元フレーム番号,パノラマセンサデータ)

		}

		//cout << n << " >= " << pano_cap.get(CV_CAP_PROP_FRAME_COUNT) << endl;

		/*
		 warpPerspective(white_img, aria1, homography, Size(PANO_W, PANO_H),
		 warpPerspective(white_img, aria1, homography, Size(PANO_W, PANO_H),
		 CV_INTER_LINEAR | CV_WARP_FILL_OUTLIERS);

		 warpPerspective(white_img, aria2, tmp_base, Size(PANO_W, PANO_H),
		 CV_INTER_LINEAR | CV_WARP_FILL_OUTLIERS);
		 bitwise_and(aria1, aria2, aria1);

		 imshow("mask", aria1);
		 waitKey(20);
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
		 */
		/*
		 // ホモグラフィー行列による投影範囲の重なりと，
		 // ホモグラフィーの類似度（差のノルム）で判断
		 // 結果としていまいちなのでコメントアウト
		 double homo_norm, min_norm = DBL_MAX;
		 homo_norm = norm(homography - tmp_base);
		 if (homo_norm < min_norm) {
		 min_norm = homo_norm;
		 detect_frame_num = n;
		 h_base = tmp_base.clone();
		 }
		 */
		//tmp_base.release();
	}

	free(sensor);
	//detect_frame_num = 143;

	// 良いものリストの中から最良のもの（最もよくマッチングがとれたもの）
	// を繰りかえし処理で見つける
	int idx = 0;                   // 反復回数のカウンタ
	std::vector<cv::DMatch> adopt; // 最良のマッチングがとれたペアでのマッチ
	size_t max_adopt = 0;          // 正しい対応点の最大数
	//cout << "sizeof near_list : " << near_senser_list.size() << endl;
	for (list<pair<int, SENSOR_DATA> >::iterator it = near_senser_list.begin();it != near_senser_list.end(); it++, ++idx) {

		vector<uchar> state;
		Mat current_near;
		Mat calced_homography = homography.clone();
		Mat current_discriptor;
		vector<KeyPoint> current_keypoints;
		vector<DMatch> current_adopt;

		////cout << "check#" << idx << endl;
		//cout << "num_of_frame : " << (*it).first << endl;

		// 該当フレーム番号のホモグラフィ行列を取り出す
		ss << "homo_" << (*it).first;
		read(node[ss.str()], tmp_base);
		ss.clear();
		ss.str("");

		ss << "keypoints_" << (*it).first;
		read(node[ss.str()], current_keypoints);
		ss.clear();
		ss.str("");

		ss << "descriptors_" << (*it).first;
		read(node[ss.str()], current_discriptor);
		ss.clear();
		ss.str("");

		// 該当フレーム番号のフレーム画像を取得(先頭に戻ってから当該フレームを取得)
		// cap.get()でシークするとdupフレームが考慮されなくなる?
		pano_cap.set(CV_CAP_PROP_POS_FRAMES, (*it).first);
		//for (int i = 0; i < (*it).first - 1; i++)
		//	pano_cap.grab();
		pano_cap >> current_near;

		// 近い背景フレーム画像を書き出し
		//ss << save_path << "near_frame_" << setw(4) << (*it).first << ".jpg";
		//imwrite(ss.str(), current_near);
		//ss.clear();
		//ss.str("");

		// 毎回特徴量検出記述をしている
		//cvtColor(current_near, gray_img2, CV_RGB2GRAY);
		//g_feature_detector.detect(gray_img2,current_keypoints);
		//feature->operator ()(gray_img2, Mat(), current_keypoints,current_discriptor, false);

/*
		 {
			Mat img32;
			gray_img2.convertTo(img32,CV_32F,1.0/255.0);
			akaze.Create_Nonlinear_Scale_Space(img32);
			akaze.Feature_Detection(current_keypoints);
			akaze.Compute_Descriptors(current_keypoints,current_discriptor);
		 }
*/
		//good_matcher(objectDescriptors, current_discriptor, &objectKeypoints,&current_keypoints, &matches, &pt1, &pt2);

		//using detail::BestOf2NearestMatcher by Opencv (優秀)
		vector<detail::ImageFeatures> features;
		features.clear();
		detail::ImageFeatures aa;
		aa.descriptors = current_discriptor.clone();
		aa.img_idx = 0;
		aa.img_size = current_near.size();
		aa.keypoints = current_keypoints;
		features.push_back(aa);

		aa.descriptors = objectDescriptors.clone();
		aa.img_idx = 1;
		aa.img_size = current_near.size();
		aa.keypoints = objectKeypoints;
		features.push_back(aa);

		Mat estA1, estA2;
		estA1 = A1Matrix.clone();
		estA2 = A2Matrix.clone();
		//cout << calced_homography << endl;

		//rotationbase estimation
		calced_homography = rotation_estimater(A1Matrix, A2Matrix, features,estA1, estA2,current_adopt);

		// using findhomography
		//calced_homography = findHomography(Mat(pt1), Mat(pt2), state,CV_RANSAC);
		//current_adopt.clear();
		//for (size_t i = 0; i < matches.size(); i++)
		//	if (state[i] == 1) {
		//		current_adopt.push_back(matches[i]);
		//	}

		//cout << "findhomography : " << endl;
		/*	cout << calced_homography << endl;
		 {
		 Mat R, K, R0, K0;
		 cameras[1].R.convertTo(R, CV_64F);
		 cameras[0].R.convertTo(R0, CV_64F);
		 Mat(cameras[1].K()).convertTo(K, CV_64F);
		 Mat(cameras[0].K()).convertTo(K0, CV_64F);
		 //cout << R.t() * K_inv << endl;;
		 cout << A1Matrix.type() << R.type() << A2Matrix.inv().type()
		 << endl;
		 calced_homography = K0 * R0.inv() * R * K.inv();

		 }
		 cout << "cameras K :" << endl;
		 cout << cameras[1].K() << endl;
		 cout << cameras[0].K() << endl;
		 */
		//cout << "rotationbased homography : " << endl;
		//cout << calced_homography << endl;
		/*
		 //using stitcher by opencv
		 // これを使うと何も考えずに2枚の画像を貼り合わせられる
		 // ただ，視線方向やパノラマのサイズ，blendの設定等があまりいじれないため
		 // 現状ではこのプログラムには組み込めない
		 // 正しく合成できるか否かの確認用
		 Mat pano;
		 vector<Mat> imgs;
		 imgs.push_back(current_near);
		 imgs.push_back(target_frame);

		 Stitcher stitcher = Stitcher::createDefault(false);
		 stitcher.setWarper(new cv::PlaneWarper());
		 stitcher.setBlender(
		 detail::Blender::createDefault(detail::Blender::NO, false));

		 Stitcher::Status status = stitcher.stitch(imgs, pano);

		 imshow("stitch test", pano);
		 waitKey(0);

		 */

		// マッチング結果を画像で保存
		drawMatches(target_frame, objectKeypoints, current_near,
				current_keypoints, current_adopt, result, Scalar::all(-1),
				Scalar::all(-1), vector<char>(), 2);

		ss << "#" << idx << " adopted size :" << current_adopt.size();
		putText(result, ss.str(), cv::Point(50, 50), FONT_HERSHEY_SIMPLEX, 1.2,cv::Scalar(0, 0, 200), 2, CV_AA);
		ss.clear();
		ss.str("");

		ss << save_path << "adopt_matches_" << setw(4) << (*it).first << ".jpg";
		imwrite(ss.str(), result);
		ss.clear();
		ss.str("");
		{
		// 現在の背景フレームを使ったパノラマ映像合成(maskは更新しない)
		warpPerspective(white_img, pano_black,	tmp_base *calced_homography, Size(PANO_W, PANO_H),	CV_INTER_LINEAR | CV_WARP_FILL_OUTLIERS);
		warpPerspective(target_frame, transform_image,tmp_base *calced_homography, Size(PANO_W, PANO_H),	CV_INTER_LINEAR | CV_WARP_FILL_OUTLIERS);
		Mat tmp_mask = mask.clone();
		Mat tmp_pano,diff_pano;
		tmp_pano = transform_image2.clone();
		make_pano(transform_image, tmp_pano, tmp_mask, pano_black);

		ss << save_path << "adopt_matches_panorama_" << setw(4) << (*it).first << ".jpg";
		imwrite(ss.str(), tmp_pano);
		ss.clear();
		ss.str("");

		// grayscaleにしてパノラマ背景とパノラマ映像フレームとの差分を計算
		Mat gray_pano1,gray_pano2,and_mask,diff_result;
		cvtColor(tmp_pano,gray_pano1,CV_BGR2GRAY);
		cvtColor(transform_image2,gray_pano2,CV_BGR2GRAY);
		absdiff(gray_pano1,gray_pano2,diff_pano);

		bitwise_and(pano_black,mask,and_mask);

		//差分画像をmask
		diff_pano.copyTo(diff_result,and_mask);

		{
			unsigned long sum=0,count=0;
			int w,h;
			#pragma omp parallel for  private(h) reduction(+:sum) reduction(+:count)
			for(w=0;w < diff_result.cols;++w){
				for(h = 0;h < diff_pano.rows;++h){
					if(diff_result.at<uchar>(h,w)!=0){
						sum+=diff_result.at<uchar>(h,w);
						++count;
					}
				}
			}
			cout << (float)sum/(float)count << "\t";
			ss << "#" << idx << " diff :" << (float)sum/(float)count;
			putText(diff_result, ss.str(), cv::Point(50, 300), FONT_HERSHEY_SIMPLEX, 5,cv::Scalar(255, 255, 255), 5, CV_AA);
			ss.clear();
			ss.str("");

			ss << save_path << "adopt_matches_panorama_diff_" << setw(4) << (*it).first << ".jpg";
			imwrite(ss.str(), diff_result);
			ss.clear();
			ss.str("");
		}

		}

		//cout << "det : " << determinant(tmp_base * calced_homography - tmp_base) << endl;
		//cout << "det : " << determinant(calced_homography) - 1.0 << endl;
		//cout << "adopted size :" << current_adopt.size() << endl;
		//cout << "adopted size :" << adopted_size << endl;
		{
			Mat samples,mean,covar;
			vector<Point2f>pt1,pt2;
			for(auto i :current_adopt){
				pt1.push_back(current_keypoints[i.trainIdx].pt);
				pt2.push_back(objectKeypoints[i.queryIdx].pt);
			}
			calcCovarMatrix(Mat(pt1).reshape(1),covar,mean,CV_COVAR_NORMAL|CV_COVAR_ROWS|CV_COVAR_SCALE,CV_32F);
			cout << covar.at<float>(0,0) << "\t" << covar.at<float>(0,1) << "\t"<< covar.at<float>(1,0) << "\t"<< covar.at<float>(1,1) << "\t";
			cout << mean.at<float>(0) << "\t" << mean.at<float>(1) << "\t";
			calcCovarMatrix(Mat(pt2).reshape(1),covar,mean,CV_COVAR_NORMAL|CV_COVAR_ROWS|CV_COVAR_SCALE,CV_32F);
			cout << covar.at<float>(0,0) << "\t" << covar.at<float>(0,1) << "\t"<< covar.at<float>(1,0) << "\t"<< covar.at<float>(1,1) << "\t";
			cout << mean.at<float>(0) << "\t" << mean.at<float>(1) << "\t" <<endl;

		}
		// 対応点が最大であるペアを採用する
		if (current_adopt.size() > max_adopt) {

			// 現状で最良のホモグラフィ行列を記録
			//cout << "find better pair" << endl;
			adopt = current_adopt;                   // 最良のペアとして更新
			max_adopt = adopt.size();				   // 正しい対応点の最大数
			homography = calced_homography.clone();  // ホモグラフィ行列を更新
			near_frame = current_near.clone();       // 背景画像を更新
			h_base = tmp_base.clone();               // 背景投影に使われたホモグラフィ行列の更新
			detect_frame_num = (*it).first;			  // 背景フレームのフレーム番号
			near_sd = (*it).second;					  // センサによるORI(回転量)
			calced_A1 = estA1.clone();
			calced_A2 = estA2.clone();
		}
	}
//	cout << "detected near frame : " << detect_frame_num << endl;
	//imwrite("best_near.jpg", near_frame);
	//cout << "h_base" << endl << h_base << endl;

	/*
	 //detect_frame_num = 101;
	 ss << "homo_" << detect_frame_num;
	 //ss << "homo_" << 3041;
	 read(node[ss.str()], tmp_base);
	 h_base = tmp_base.clone();
	 ss.clear();
	 ss.str("");
	 pano_cap.set(CV_CAP_PROP_POS_FRAMES, 0);
	 for (int i = 0; i < detect_frame_num + 1; i++)
	 pano_cap >> near_frame;

	 imwrite("near_frame.jpg", near_frame);

	 //imshow("base frame",near_frame);
	 // 合成対象のフレームと最も近いフレームパノラマ背景のフレームのチャネルごとのヒストグラムを計算
	 if (f_adj_color != 0) {
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

	 LUT(near_channels[i], Mat(Size(256, 1), CV_8U, lut),
	 tmp_channel);
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
	 //LUT(transform_image2, Mat(Size(256, 1), CV_8U, lut), result);
	 //transform_image2 = result.clone();
	 near_frame = fix_near_image.clone();
	 //imwrite("fix_test.jpg", transform_image2);
	 }*/
	/*
	 /*
	 *  ハフ変換を用いて直線を検出し，特徴点を検出する際のマスクとして使う
	 *
	 *
	 Mat hough_src, hough_dst; // ハフ変換の入力と，検出した線の出力先
	 cvtColor(near_frame, gray_img2, CV_RGB2GRAY);
	 Canny(gray_img2, hough_src, 30, 50, 3);

	 imshow("detected lines", hough_src);

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
	 */
	/*	for (int i = 0; i < hough_dst.cols; i++)
	 for (int j = 0; j < hough_dst.rows; j++) {
	 if (i > hough_dst.cols / 2.5 || i < hough_dst.cols / 8)
	 hough_dst.at<unsigned char> (j, i) = 0;
	 if (j < hough_dst.rows / 8 || j > hough_dst.rows * 3.0 / 4.0)
	 hough_dst.at<unsigned char> (j, i) = 0;
	 }
	 */

	/*
	 // 視線方向が近いフレーム画像から特徴点を抽出しマッチング
	 //dist_src = near_frame.clone();
	 //undistort(dist_src,near_frame,A1Matrix,dist);
	 Mat half_near_frame; // 高速化のために1/2解像度で特徴点検出

	 //resize(near_frame,half_near_frame,Size(near_frame.cols/2.0,near_frame.rows/2.0));
	 //resize(near_frame,near_frame,Size(near_frame.cols/2.0,near_frame.rows/2.0));
	 //1/2解像度のグレースケース画像を生成
	 cvtColor(near_frame, gray_img2, CV_RGB2GRAY);
	 //cvtColor(half_near_frame, gray_img2, CV_RGB2GRAY);

	 // 特徴点抽出範囲のマスク画像
	 Mat feature_mask = Mat::zeros(near_frame.size(), CV_8U);
	 for (int i = 0; i < feature_mask.cols; i++)
	 for (int j = 0; j < feature_mask.rows; j++)
	 if (i > 600 && i < 1100 && j > 250 && j < 650)
	 feature_mask.at<uchar>(j, i) = 255;

	 imwrite("featuer_mask.jpg", feature_mask);

	 //g_feature_detector.detect(gray_img2,imageKeypoints);
	 feature->operator ()(gray_img2, Mat(), imageKeypoints, imageDescriptors);

	 good_matcher(objectDescriptors, imageDescriptors, &objectKeypoints,
	 &imageKeypoints, &matches, &pt1, &pt2);
	 drawMatches(target_frame, objectKeypoints, near_frame, imageKeypoints,
	 matches, result);

	 cout << "show matches" << endl;
	 imwrite("matching.jpg", result);

	 Mat surf_img = target_frame.clone();

	 for (int i = 0; i < objectKeypoints.size(); i++) {
	 bool flag = false;
	 for (int j = 0; j < matches.size(); j++) {
	 if (matches[j].queryIdx == i)
	 flag = true;

	 }
	 if (flag == false)
	 continue;
	 KeyPoint* point = &(objectKeypoints[i]);
	 Point center; // Key Point's Center
	 int radius; // Radius of Key Point
	 center.x = cvRound(point->pt.x);
	 center.y = cvRound(point->pt.y);
	 radius = cvRound(point->size * 0.25);
	 circle(surf_img, center, radius, Scalar(255, 255, 0), 1, 8, 0);
	 }
	 imwrite("img1.jpg", surf_img);

	 surf_img = near_frame.clone();
	 for (int i = 0; i < imageKeypoints.size(); i++) {
	 bool flag = false;
	 for (int j = 0; j < matches.size(); j++) {
	 if (matches[j].trainIdx == i)
	 flag = true;

	 }
	 if (flag == false)
	 continue;
	 KeyPoint* point = &(imageKeypoints[i]);
	 Point center; // Key Point's Center
	 int radius; // Radius of Key Point
	 center.x = cvRound(point->pt.x);
	 center.y = cvRound(point->pt.y);
	 radius = cvRound(point->size * 0.25);
	 circle(surf_img, center, radius, Scalar(255, 255, 0), 1, 8, 0);
	 }
	 imwrite("img2.jpg", surf_img);

	 // ホモグラフィ行列を計算し射影変換する
	 vector<uchar> state;
	 homography = findHomography(Mat(pt1), Mat(pt2), state, CV_RANSAC, 10.0);

	 std::vector<cv::DMatch> adopt;
	 for (int i = 0; i < matches.size(); i++)
	 if (state[i] == 1)
	 adopt.push_back(matches[i]);

	 drawMatches(target_frame, objectKeypoints, near_frame, imageKeypoints,
	 adopt, result);

	 imwrite("adopt_matches.jpg", result);

	 */
	/*
	 list<pair<int, SENSOR_DATA> >::iterator it = near_senser_list.begin();
	 aim_sd = (*++it).second;

	 SetYawRotationMatrix(&yawMatrix, (pano_sd.alpha - aim_sd.alpha));
	 SetPitchRotationMatrix(&pitchMatrix, (pano_sd.beta - aim_sd.beta));
	 SetRollRotationMatrix(&rollMatrix, (pano_sd.gamma - aim_sd.gamma));
	 senserbased_homographty = homography.clone();
	 senserbased_homographty = A1Matrix * rollMatrix * pitchMatrix * yawMatrix
	 * A1Matrix.inv();
	 */
	if (h_base.empty() || homography.empty())
		cerr << "h_base or homography is empty" << endl;
	/*
	 vector<KeyPoint> masked_vec1, masked_vec2;
	 Mat masked_mat1 = Mat(objectDescriptors.rows - adopt.size(),
	 objectDescriptors.cols, objectDescriptors.type());
	 Mat masked_mat2 = Mat(imageDescriptors.rows - adopt.size(),
	 imageDescriptors.cols, imageDescriptors.type());

	 int c1 = 0;
	 int c2 = 0;
	 bool flag = false;
	 // 先のペアで採用されてたマッチングを除いたキーポイントを作る
	 for (int i = 0; i < objectDescriptors.rows; i++) {
	 flag = false;
	 for (int j = 0; j < adopt.size(); j++) {
	 if (adopt[j].queryIdx == i) {
	 flag = true;
	 }
	 }
	 if (!flag) {
	 masked_vec1.push_back(objectKeypoints[i]);
	 objectDescriptors.row(i).copyTo(masked_mat1.row(c1));
	 ++c1;
	 }

	 }

	 for (int i = 0; i < imageDescriptors.rows; i++) {
	 flag = false;
	 for (int j = 0; j < adopt.size(); j++) {
	 if (adopt[j].trainIdx == i) {
	 flag = true;
	 }
	 }
	 if (!flag) {
	 masked_vec2.push_back(imageKeypoints[i]);
	 imageDescriptors.row(i).copyTo(masked_mat2.row(c2));
	 ++c2;
	 }

	 }
	 cout << "test" << endl;
	 cout << masked_vec1.size() << endl;
	 cout << "rows cols : " << masked_mat1.rows << " " << masked_mat1.cols
	 << endl;

	 cout << masked_vec2.size() << endl;
	 cout << "rows cols : " << masked_mat2.rows << " " << masked_mat2.cols
	 << endl;

	 vector<DMatch> re_matches;
	 vector<Point2f> re_pt1, re_pt2;
	 good_matcher(masked_mat1, masked_mat2, &masked_vec1, &masked_vec2,
	 &re_matches, &re_pt1, &re_pt2);

	 //	vector<Point2f> re_pt1,re_pt2;
	 // for (int i = 0; i < re_matches.size(); i++) {
	 // re_pt1.push_back(masked_vec1[re_matches[i].queryIdx].pt);
	 // re_pt2.push_back(masked_vec2[re_matches[i].trainIdx].pt);
	 // }

	 drawMatches(target_frame, masked_vec1, near_frame, masked_vec2, re_matches,
	 result);

	 vector<uchar> re_state;
	 Mat re_homography = findHomography(Mat(re_pt1), Mat(re_pt2), re_state,
	 CV_RANSAC, 10.0);



	 imshow("remutch_test", result);
	 waitKey();
	 imwrite("remutch_test.jpg", result);


	 vector<DMatch> re_adopt;
	 for (int i = 0; i < re_matches.size(); i++)
	 if (re_state[i] == 1)
	 re_adopt.push_back(re_matches[i]);

	 drawMatches(target_frame, masked_vec1, near_frame, masked_vec2,re_adopt, result);



	 imshow("readopt_test", result);
	 waitKey();
	 imwrite("readopt_test.jpg", result);
	 */
	SetYawRotationMatrix(&yawMatrix, near_sd.alpha);
	SetPitchRotationMatrix(&pitchMatrix, near_sd.beta);
	SetRollRotationMatrix(&rollMatrix, near_sd.gamma);
	cout << " near_sd yaw : " << near_sd.alpha << " pitch : " << near_sd.beta
			<< " roll : " << near_sd.gamma << endl;
	sensing_rotaion1 = rollMatrix* pitchMatrix * yawMatrix ;

	warpPerspective(white_img, pano_black,	calced_A2 * sensing_rotaion1.inv() * sensing_rotaion2* calced_A1.inv(), Size(PANO_W, PANO_H),	CV_INTER_LINEAR | CV_WARP_FILL_OUTLIERS);
	warpPerspective(target_frame, transform_image,calced_A2 * sensing_rotaion1.inv() * sensing_rotaion2* calced_A1.inv(), Size(PANO_W, PANO_H),	CV_INTER_LINEAR | CV_WARP_FILL_OUTLIERS);
	Mat tmp = Mat(Size(PANO_W, PANO_H), CV_8UC3, Scalar::all(0));

	Mat tmp_ori = Mat(tmp, Rect(0, 0, 1280, 720));
	near_frame.copyTo(tmp_ori);
	imwrite("senser_pano1.jpg", tmp);
	make_pano(transform_image, tmp, mask, pano_black);

	imwrite("senser_pano2.jpg", tmp);
	{
		double yaw, pitch, roll;
		Mat rotation = calced_A2.inv() * homography * calced_A1;
		roll = std::atan2(rotation.at<double>(1, 0), rotation.at<double>(1, 1));
		pitch = std::asin(rotation.at<double>(2, 1));
		yaw = std::atan2(rotation.at<double>(0, 2), rotation.at<double>(2, 2));


		roll = roll / CV_PI * 180.0;
		pitch = pitch / CV_PI * 180.0;
		yaw = yaw / CV_PI * 180.0;
		//if (rotation.at<double>(2, 2) < 0.0)
		//	yaw = 180.0 - yaw;

		std::cout << "LevMarq Rotation euler\n" << "yaw : " << yaw
				<< " pitch : " << pitch << " roll : " << roll << std::endl;
		std::cout << "senser Rotation euler\n" << "yaw : "
				<< -near_sd.alpha + aim_sd.alpha << " pitch : "
				<< -near_sd.beta + aim_sd.beta << " roll : "
				<< -near_sd.gamma + aim_sd.gamma << std::endl;
	}

	warpPerspective(white_img, pano_black, h_base * homography,	Size(PANO_W, PANO_H), CV_INTER_LINEAR | CV_WARP_FILL_OUTLIERS);
	warpPerspective(target_frame, transform_image, h_base * homography,Size(PANO_W, PANO_H), CV_INTER_LINEAR | CV_WARP_FILL_OUTLIERS);

	// ホモグラフィ行列を記録
	for (int ii = 1; ii < target_frame_num; ii++) {
		if (target_frame_num < ii * 3) {
			target_frame_num = ii;
			break;
		}
	}
	ss << save_path << "homography-" << setw(4) << target_frame_num << ".xml";

	FileStorage fs(ss.str(), cv::FileStorage::WRITE);
	ss.clear();
	ss.str("");

	// パノラマ平面の内部パラメータを計算()
	Mat panorama_in = A1Matrix.clone();
	panorama_in.at<double>(0, 0) = A1Matrix.at<double>(0, 0);
	panorama_in.at<double>(1, 1) = A1Matrix.at<double>(1, 1);
	panorama_in.at<double>(0, 2) = PANO_W / 2.0;
	panorama_in.at<double>(1, 2) = PANO_H / 2.0;

	write(fs, "H_Base", h_base);
	write(fs, "Panorama_Homography", Mat(h_base * homography));
	write(fs, "Homography", homography);
	write(fs, "Panorama_Rotaton",Mat(panorama_in.inv() * h_base * homography * calced_A1));
	write(fs, "Rotaton",Mat(calced_A2.inv() * homography * calced_A1));
	write(fs, "A1", calced_A1);
	write(fs, "A2", calced_A2);
	write(fs, "A_Panorama", panorama_in);
	write(fs, "BackgroundFrameNum",(int)detect_frame_num);

	//write(fs, "hbasehomography", Mat(h_base * homography));
	//write(fs, "homography", homography);
	//write(fs, "sensing_rotaion1Xsensing_rotation2T",
	//		Mat(sensing_rotaion2.inv() * sensing_rotaion1));
	//write(fs, "sensing_rotaion1", sensing_rotaion1);
	//write(fs, "sensing_rotaion2", sensing_rotaion2);
	//write(fs, "LevMarq_rotaton", Mat(calced_A2.inv() * homography * calced_A1));

	//imshow("transform_image", transform_image);
	//waitKey(20);

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

	Mat tmp_result;
	mask2 = mask.clone();
	bitwise_and(mask, pano_black, mask2);

	//imshow("panoblack", pano_black);
	//waitKey(30);
	make_pano(transform_image, transform_image2, mask, pano_black);
	ss.str("");
	ss.clear();
	ss << save_path << result_name;
	imwrite(ss.str(), transform_image2);

	// TODO :回転の微調整をする
	/*
	 int count = 0;
	 for (double alpha = -2.0; alpha < 2; alpha += 0.5) {
	 for (double beta = -2.0; beta < 2; beta += 0.5) {
	 for (double gamma = -2.0; gamma < 2; gamma += 0.5) {
	 SetYawRotationMatrix(&yawMatrix, alpha);
	 SetPitchRotationMatrix(&pitchMatrix, beta);
	 SetRollRotationMatrix(&rollMatrix, gamma);
	 //SetYawRotationMatrix(&yawMatrix, 0);
	 //SetPitchRotationMatrix(&pitchMatrix, 0);
	 //SetRollRotationMatrix(&rollMatrix, 0);
	 Mat adj_tmp = h_base * homography * A1Matrix * rollMatrix
	 * pitchMatrix * yawMatrix * A1Matrix.inv();
	 warpPerspective(white_img, pano_black, adj_tmp,
	 Size(PANO_W, PANO_H),
	 CV_INTER_LINEAR | CV_WARP_FILL_OUTLIERS);
	 warpPerspective(target_frame, transform_image, adj_tmp,
	 Size(PANO_W, PANO_H),
	 CV_INTER_LINEAR | CV_WARP_FILL_OUTLIERS);
	 bitwise_and(mask, pano_black, mask2);
	 tmp_result = panorama.clone();
	 make_pano(transform_image, tmp_result, ~mask2, pano_black);
	 ss.str("");
	 ss.clear();
	 ss << "test_" << count << ".jpg";
	 imwrite(ss.str(), tmp_result);

	 count++;
	 }
	 }
	 }
	 */
	//imshow("result", transform_image2);
	//waitKey(0);
	gettimeofday(&t1, NULL);

	ofstream ofs("processing_time.txt");
	if (algorithm_type.compare("SURF") == 0)
		ofs << "<algo>" << "\n" << "SURF" << endl;
	else if (algorithm_type.compare("SIFT") == 0)
		ofs << "<algo>" << "\n" << "SIFT" << endl;

	ofs << local->tm_year + 1900 << "/" << local->tm_mon + 1 << "/"
			<< local->tm_mday << " " << local->tm_hour << ":" << local->tm_min
			<< ":" << local->tm_sec << " " << local->tm_isdst << endl;
	ofs << "<start>" << "\n" << t0.tv_sec << "." << t0.tv_usec << "[sec]"
			<< endl;
	ofs << "<end>" << "\n" << t1.tv_sec << "." << t1.tv_usec << "[sec]" << endl;
	if (t1.tv_usec < t0.tv_usec)
		ofs << "<processing_time>" << "\n" << t1.tv_sec - t0.tv_sec - 1.0 << "."
				<< 1000000 + t1.tv_usec - t0.tv_usec << "[sec]" << endl;
	else
		ofs << "<processing_time>" << "\n" << t1.tv_sec - t0.tv_sec << "."
				<< t1.tv_usec - t0.tv_usec << "[sec]" << endl;

	cout << "finished" << endl;
	return 0;
}
