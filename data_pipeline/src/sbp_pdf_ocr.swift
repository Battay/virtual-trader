import Foundation
import Vision

guard CommandLine.arguments.count >= 2 else {
    fputs("Usage: sbp_pdf_ocr.swift IMAGE [IMAGE ...]\n", stderr)
    exit(2)
}

for imagePath in CommandLine.arguments.dropFirst() {
    let request = VNRecognizeTextRequest()
    request.recognitionLevel = .accurate
    request.usesLanguageCorrection = true
    request.recognitionLanguages = ["en-US"]
    let handler = VNImageRequestHandler(
        url: URL(fileURLWithPath: imagePath),
        options: [:]
    )
    do {
        try handler.perform([request])
    } catch {
        fputs("Vision OCR failed: \(error)\n", stderr)
        exit(6)
    }
    let observations = (request.results ?? []).sorted {
        let firstY = $0.boundingBox.midY
        let secondY = $1.boundingBox.midY
        if abs(firstY - secondY) > 0.005 {
            return firstY > secondY
        }
        return $0.boundingBox.minX < $1.boundingBox.minX
    }
    for observation in observations {
        if let candidate = observation.topCandidates(1).first {
            print(candidate.string)
        }
    }
}
