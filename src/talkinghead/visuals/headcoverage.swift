// Person-mask coverage inside rectangles, via Apple Vision.
//
// The visuals head-clip check feeds one JSON request per cook on stdin and
// reads one JSON response on stdout. Rects are in each image's own pixel
// coordinates; coverage is the fraction of person-segmentation pixels
// (>= 50% alpha) inside the rect. Person segmentation covers hair and body
// silhouette, which face-box detectors miss on profile shots.

import Foundation
import Vision
import CoreImage

struct Item: Decodable { let path: String; let rect: [Double] }
struct Req: Decodable { let items: [Item] }
struct Res: Encodable { let path: String; let coverage: Double; let error: String? }

let data = FileHandle.standardInput.readDataToEndOfFile()
let req: Req
do {
    req = try JSONDecoder().decode(Req.self, from: data)
} catch {
    fputs("headcoverage: bad request JSON: \(error)\n", stderr)
    exit(2)
}

let context = CIContext()
var results: [Res] = []

for item in req.items {
    do {
        guard let image = CIImage(contentsOf: URL(fileURLWithPath: item.path)) else {
            throw NSError(domain: "headcoverage", code: 1,
                          userInfo: [NSLocalizedDescriptionKey: "cannot read \(item.path)"])
        }
        let handler = VNImageRequestHandler(ciImage: image, options: [:])
        let seg = VNGeneratePersonSegmentationRequest()
        seg.qualityLevel = .accurate
        seg.outputPixelFormat = kCVPixelFormatType_OneComponent8
        try handler.perform([seg])
        guard let obs = seg.results?.first as? VNPixelBufferObservation else {
            throw NSError(domain: "headcoverage", code: 2,
                          userInfo: [NSLocalizedDescriptionKey: "no segmentation result for \(item.path)"])
        }
        let mask = obs.pixelBuffer
        let mw = CVPixelBufferGetWidth(mask)
        let mh = CVPixelBufferGetHeight(mask)
        // Vision may return the mask at a reduced resolution; scale the rect.
        let iw = Double(image.extent.width)
        let ih = Double(image.extent.height)
        let sx = Double(mw) / iw
        let sy = Double(mh) / ih
        let x0 = max(0, Int((item.rect[0] * sx).rounded(.down)))
        let y0 = max(0, Int((item.rect[1] * sy).rounded(.down)))
        let x1 = min(mw, Int(((item.rect[0] + item.rect[2]) * sx).rounded(.up)))
        let y1 = min(mh, Int(((item.rect[1] + item.rect[3]) * sy).rounded(.up)))

        var inside = 0
        var total = 0
        CVPixelBufferLockBaseAddress(mask, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(mask, .readOnly) }
        if let base = CVPixelBufferGetBaseAddress(mask) {
            let stride = CVPixelBufferGetBytesPerRow(mask)
            let buf = base.assumingMemoryBound(to: UInt8.self)
            if x1 > x0 && y1 > y0 {
                for y in y0..<y1 {
                    let row = buf + y * stride
                    for x in x0..<x1 {
                        total += 1
                        if row[x] >= 128 { inside += 1 }
                    }
                }
            }
        }
        let coverage = total > 0 ? Double(inside) / Double(total) : 0.0
        results.append(Res(path: item.path, coverage: coverage, error: nil))
    } catch {
        results.append(Res(path: item.path, coverage: 0.0, error: String(describing: error)))
    }
}

do {
    struct Resp: Encodable { let results: [Res] }
    let out = try JSONEncoder().encode(Resp(results: results))
    FileHandle.standardOutput.write(out)
    FileHandle.standardOutput.write(Data("\n".utf8))
} catch {
    fputs("headcoverage: cannot encode response: \(error)\n", stderr)
    exit(3)
}
