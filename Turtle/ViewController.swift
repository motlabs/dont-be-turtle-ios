//
//  ViewController.swift
//  PoseEstimation-CoreML
//
//  Created by GwakDoyoung on 05/07/2018.
//  Copyright © 2018 tucan9389. All rights reserved.
//

import UIKit
import Vision
import CoreMedia

class ViewController: UIViewController, VideoCaptureDelegate {
    
    public typealias BodyPoint = (point: CGPoint, confidence: Double)
    public typealias DetectObjectsCompletion = ([BodyPoint?]?, Error?) -> Void
    
    // MARK: - UI 프로퍼티
    
    @IBOutlet weak var videoPreview: UIView!
    @IBOutlet weak var poseView: PoseView!
    @IBOutlet weak var labelsTableView: UITableView!
    
    @IBOutlet weak var inferenceLabel: UILabel!
    @IBOutlet weak var etimeLabel: UILabel!
    @IBOutlet weak var fpsLabel: UILabel!
    
    @IBOutlet weak var turtleBGView: UIView!
    @IBOutlet weak var angleLabel: UILabel!
    
    private var tableData: [BodyPoint?] = []
    
    private var mvFilter: MovingAverageFilter = MovingAverageFilter()
    private var mvFilterAngle: MovingAverageFilterAngle = MovingAverageFilterAngle()
    
    // MARK - 성능 측정 프러퍼티
    private let 👨‍🔧 = 📏()
    
    
    // MARK - Core ML model
    typealias EstimationModel = model_40500
    var coremlModel: EstimationModel? = nil
    // mv2_cpm_1_three
    
    
    
    
    // MARK: - Vision 프로퍼티
    
    var request: VNCoreMLRequest!
    var visionModel: VNCoreMLModel! {
        didSet {
            request = VNCoreMLRequest(model: visionModel, completionHandler: visionRequestDidComplete)
            request.imageCropAndScaleOption = .centerCrop
        }
    }
    
    
    // MARK: - AV 프로퍼티
    
    var videoCapture: VideoCapture!
    let semaphore = DispatchSemaphore(value: 2)
    
    
    // MARK: - 라이프사이클 메소드
    
    override func viewDidLoad() {
        super.viewDidLoad()
        
//        self.title = "lsp chk 370000"
        
        // MobileNet 클래스는 `MobileNet.mlmodel`를 프로젝트에 넣고, 빌드시키면 자동으로 생성된 랩퍼 클래스
        // MobileNet에서 만든 model: MLModel 객체로 (Vision에서 사용할) VNCoreMLModel 객체를 생성
        // Vision은 모델의 입력 크기(이미지 크기)에 따라 자동으로 조정해 줌
        visionModel = try? VNCoreMLModel(for: EstimationModel().model)
        
        // 카메라 세팅
        setUpCamera()
        
        // 레이블 테이블 세팅
        labelsTableView.dataSource = self
        
        // 레이블 점 세팅
        poseView.setUpOutputComponent()
        
        // 성능측정용 델리게이트 설정
        👨‍🔧.delegate = self
    }
    
    override func didReceiveMemoryWarning() {
        super.didReceiveMemoryWarning()
    }
    
    
    // MARK: - 초기 세팅
    
    func setUpCamera() {
        videoCapture = VideoCapture()
        videoCapture.delegate = self
        videoCapture.fps = 30
        videoCapture.setUp(sessionPreset: .vga640x480) { success in
            
            if success {
                // UI에 비디오 미리보기 뷰 넣기
                if let previewLayer = self.videoCapture.previewLayer {
                    self.videoPreview.layer.addSublayer(previewLayer)
                    self.resizePreviewLayer()
                }
                
                // 초기설정이 끝나면 라이브 비디오를 시작할 수 있음
                self.videoCapture.start()
            }
        }
    }
    
    func resizePreviewLayer() {
        videoCapture.previewLayer?.frame = videoPreview.bounds
    }
    
    
    
    // MARK: - 추론하기
    
    func predictUsingVision(pixelBuffer: CVPixelBuffer) {
        // Vision이 입력이미지를 자동으로 크기조정을 해줄 것임.
        let handler = VNImageRequestHandler(cvPixelBuffer: pixelBuffer)
        try? handler.perform([request])
    }
    
    func visionRequestDidComplete(request: VNRequest, error: Error?) {
        self.👨‍🔧.🏷(with: "endInference")
        if let observations = request.results as? [VNCoreMLFeatureValueObservation],
            let heatmap = observations.first?.featureValue.multiArrayValue {
            
            // convert heatmap to [keypoint]
            let n_kpoints = convert(heatmap: heatmap)
            
            //
            self.mvFilter.addKeypoints(keypoints: n_kpoints)
            
//            let filetered_kpoints = self.mvFilter.keypoints
            let filetered_kpoints = n_kpoints
            
            DispatchQueue.main.sync {
                // draw line
                self.poseView.bodyPoints = filetered_kpoints
                
                // show key points description
                self.showKeypointsDescription(with: n_kpoints)
                
                self.turtleBGView.alpha = 0.6
                if let p1: CGPoint = filetered_kpoints[0]?.point,
                    let p2: CGPoint = filetered_kpoints[2]?.point,
                    let p3: CGPoint = filetered_kpoints[1]?.point {
                    let result: Double = Double.radianAngle(p1: p1, p2: p2, p3: p3)
                    mvFilterAngle.addAngle(newAngle: result)
                    let angle = mvFilterAngle.angle
                    self.angleLabel.text = "\(String(format: "%.1f", angle))"
                    if abs(angle) > 284 {
                        self.turtleBGView.backgroundColor = .red
                    } else {
                        self.turtleBGView.backgroundColor = .green
                    }
                }
                
                // end of measure
                self.👨‍🔧.🎬🤚()
            }
        }
    }
    
    func convert(heatmap: MLMultiArray) -> [BodyPoint?] {
        guard heatmap.shape.count >= 3 else {
            print("heatmap's shape is invalid. \(heatmap.shape)")
            return []
        }
        let keypoint_number = heatmap.shape[0].intValue
        let heatmap_w = heatmap.shape[1].intValue
        let heatmap_h = heatmap.shape[2].intValue
        
        var n_kpoints = (0..<keypoint_number).map { _ -> BodyPoint? in
            return nil
        }
        
        for k in 0..<keypoint_number {
            for i in 0..<heatmap_w {
                for j in 0..<heatmap_h {
                    let index = k*(heatmap_w*heatmap_h) + i*(heatmap_h) + j
                    let confidence = heatmap[index].doubleValue
                    guard confidence > 0 else { continue }
                    if n_kpoints[k] == nil ||
                        (n_kpoints[k] != nil && n_kpoints[k]!.confidence < confidence) {
                        n_kpoints[k] = (CGPoint(x: CGFloat(j), y: CGFloat(i)), confidence)
                    }
                }
            }
        }
        
        // transpose to (1.0, 1.0)
        n_kpoints = n_kpoints.map { kpoint -> BodyPoint? in
            if let kp = kpoint {
                return (CGPoint(x: 1 - kp.point.x/CGFloat(heatmap_w),
                                y: kp.point.y/CGFloat(heatmap_h)),
                        kp.confidence)
            } else {
                return nil
            }
        }
        
        return n_kpoints
    }
    
    func showKeypointsDescription(with n_kpoints: [BodyPoint?]) {
        self.tableData = n_kpoints
        self.labelsTableView.reloadData()
    }
    
    
    // MARK: - VideoCaptureDelegate
    
    func videoCapture(_ capture: VideoCapture, didCaptureVideoFrame pixelBuffer: CVPixelBuffer?, timestamp: CMTime) {
        // 카메라에서 캡쳐된 화면은 pixelBuffer에 담김.
        // Vision 프레임워크에서는 이미지 대신 pixelBuffer를 바로 사용 가능
        if let pixelBuffer = pixelBuffer {
            // start of measure
            self.👨‍🔧.🎬👏()
            
            // predict!
            self.predictUsingVision(pixelBuffer: pixelBuffer)
        }
    }
}

// #MARK: - UITableView 데이터소스

extension ViewController: UITableViewDataSource {
    func tableView(_ tableView: UITableView, numberOfRowsInSection section: Int) -> Int {
        return tableData.count// > 0 ? 1 : 0
    }
    
    func tableView(_ tableView: UITableView, cellForRowAt indexPath: IndexPath) -> UITableViewCell {
        let cell: UITableViewCell = tableView.dequeueReusableCell(withIdentifier: "LabelCell", for: indexPath)
        cell.textLabel?.text = Constant.pointLabels[indexPath.row]
        if let body_point = tableData[indexPath.row] {
            let pointText: String = "\(String(format: "%.3f", body_point.point.x)), \(String(format: "%.3f", body_point.point.y))"
            cell.detailTextLabel?.text = "(\(pointText)), [\(String(format: "%.3f", body_point.confidence))]"
        } else {
            cell.detailTextLabel?.text = "N/A"
        }
        return cell
    }
}


// #MARK: - 📏 델리게이트

extension ViewController: 📏Delegate {
    func updateMeasure(inferenceTime: Double, executionTime: Double, fps: Int) {
        //print(executionTime, fps)
        self.inferenceLabel.text = "inference: \(Int(inferenceTime*1000.0)) mm"
        self.etimeLabel.text = "execution: \(Int(executionTime*1000.0)) mm"
        self.fpsLabel.text = "fps: \(fps)"
    }
}


// #MARK: - 상수

struct Constant {
    static let pointLabels = [
        "top\t\t\t", //0
        "neck\t\t", //1
        
        "R shoulder\t", //2
        "R elbow\t\t", //3
        "R wrist\t\t", //4
        "L shoulder\t", //5
        "L elbow\t\t", //6
        "L wrist\t\t", //7
        
        "R hip\t\t", //8
        "R knee\t\t", //9
        "R ankle\t\t", //10
        "L hip\t\t", //11
        "L knee\t\t", //12
        "L ankle\t\t", //13
    ]
    
    static let connectingPointIndexs: [(Int, Int)] = [
        (0, 1), // top-neck
        
        (1, 2), // neck-rshoulder
        (2, 3), // rshoulder-relbow
        (3, 4), // relbow-rwrist
        (1, 8), // neck-rhip
        (8, 9), // rhip-rknee
        (9, 10), // rknee-rankle
        
        (1, 5), // neck-lshoulder
        (5, 6), // lshoulder-lelbow
        (6, 7), // lelbow-lwrist
        (1, 11), // neck-lhip
        (11, 12), // lhip-lknee
        (12, 13), // lknee-lankle
    ]
    static let jointLineColor: UIColor = UIColor(displayP3Red: 87.0/255.0,
                                                 green: 255.0/255.0,
                                                 blue: 211.0/255.0,
                                                 alpha: 0.5)
    
    static let colors: [UIColor] = [
        .red,
        .green,
        .blue,
        .cyan,
        .yellow,
        .magenta,
        .orange,
        .purple,
        .brown,
        .black,
        .darkGray,
        .lightGray,
        .white,
        .gray,
        ]
}

//struct <#name#> {
//    <#fields#>
//}

class MovingAverageFilter {
    typealias BodyPoint = ViewController.BodyPoint
    
    let maximumCount: Int = 8
    var keypointsArray: [[BodyPoint?]] = []
    func addKeypoints(keypoints: [BodyPoint?]) {
        keypointsArray.append(keypoints)
        if keypointsArray.count > maximumCount {
            keypointsArray.remove(at: 0)
        }
    }
    
    var keypoints: [BodyPoint?] {
        if let firstKeypoints = keypointsArray.first {
            var result: [BodyPoint?] = Array<BodyPoint?>(repeating: nil, count: firstKeypoints.count)
            for i in 0..<firstKeypoints.count {
                var count: Double = 0
                for j in 0..<keypointsArray.count {
                    if let kp: BodyPoint = keypointsArray[j][i] {
                        count += 1
                        if let oldkp: BodyPoint = result[i] {
                            let p = oldkp.point
                            let c = oldkp.confidence
                            result[i]?.point = CGPoint(x: p.x + kp.point.x, y: p.y + kp.point.y)
                            result[i]?.confidence = c + kp.confidence
                        } else {
                            result[i] = kp
                        }
                    }
                }
                if let kp = result[i] {
                    result[i]?.point = CGPoint(x: kp.point.x / CGFloat(count), y: kp.point.y / CGFloat(count))
                    result[i]?.confidence = kp.confidence / count
                }
            }
            return result
        } else {
            return []
        }
    }
}

class MovingAverageFilterAngle {
    
    let maximumCount: Int = 8
    var angles: [Double] = []
    func addAngle(newAngle: Double) {
        angles.append(newAngle)
        if angles.count > maximumCount {
            angles.remove(at: 0)
        }
    }
    var angle: Double {
        guard angles.count > 0 else { return 0 }
        return (angles.reduce(0) { $0 + $1 }) / Double(angles.count)
    }
}


extension Double {
    static func angle(p1: CGPoint, p2: CGPoint, p3: CGPoint) -> Double {
        let angle1: Double = Double(atan2(p1.y - p3.y, p1.x - p3.x));
        let angle2: Double = Double(atan2(p2.y - p3.y, p2.x - p3.x));
        
        return angle1 - angle2;
    }
    static func radianAngle(p1: CGPoint, p2: CGPoint, p3: CGPoint) -> Double {
        return (angle(p1: p1, p2: p2, p3: p3) / pi) * 180
    }
}
