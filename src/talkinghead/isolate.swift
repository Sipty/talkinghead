// Offline speech isolation through Apple's AUSoundIsolation (Voice Isolation) audio unit.
//
// Usage: talkinghead-isolate IN.wav OUT.wav TAIL_FRAMES
//        talkinghead-isolate --probe
//
// Renders IN.wav through the unit at 100% wet and writes OUT.wav as 32-bit float with
// (input frames + TAIL_FRAMES) frames. The unit's latency appears at the head of OUT.wav;
// the caller measures it and trims it, so nothing here assumes a value.
// Exit status: 0 success, 1 render failure, 2 usage, 3 the unit is not installed.
import AVFoundation
import Foundation

func fail(_ message: String, _ status: Int32) -> Never {
    FileHandle.standardError.write((message + "\n").data(using: .utf8)!)
    exit(status)
}

var desc = AudioComponentDescription(
    componentType: kAudioUnitType_Effect,
    componentSubType: 0x766f6973, // 'vois'
    componentManufacturer: kAudioUnitManufacturer_Apple,
    componentFlags: 0,
    componentFlagsMask: 0)
guard AudioComponentFindNext(nil, &desc) != nil else {
    fail("AUSoundIsolation (aufx 'vois' 'appl') is not installed on this machine", 3)
}

let args = CommandLine.arguments
if args.count == 2 && args[1] == "--probe" { exit(0) }
guard args.count == 4, let tail = Int64(args[3]), tail >= 0 else {
    fail("usage: talkinghead-isolate IN.wav OUT.wav TAIL_FRAMES", 2)
}

// The output file is closed when this returns, so its header is complete before the caller reads it.
func isolate(input: URL, output: URL, tail: Int64) throws {
    let inFile = try AVAudioFile(forReading: input)
    let format = inFile.processingFormat
    let total = inFile.length + tail

    let engine = AVAudioEngine()
    let player = AVAudioPlayerNode()
    let iso = AVAudioUnitEffect(audioComponentDescription: desc)
    engine.attach(player)
    engine.attach(iso)
    engine.connect(player, to: iso, format: format)
    engine.connect(iso, to: engine.mainMixerNode, format: format)
    try engine.enableManualRenderingMode(.offline, format: format, maximumFrameCount: 4096)
    try engine.start()
    player.scheduleFile(inFile, at: nil)
    player.play()

    // The input's file settings: the output keeps its sample format.
    let outFile = try AVAudioFile(forWriting: output, settings: inFile.fileFormat.settings)
    let buffer = AVAudioPCMBuffer(pcmFormat: engine.manualRenderingFormat, frameCapacity: 4096)!
    FileHandle.standardError.write("latency \(iso.auAudioUnit.latency) s\n".data(using: .utf8)!)
    while engine.manualRenderingSampleTime < total {
        let n = AVAudioFrameCount(min(Int64(4096), total - engine.manualRenderingSampleTime))
        let status = try engine.renderOffline(n, to: buffer)
        guard status == .success else { fail("render status \(status.rawValue)", 1) }
        try outFile.write(from: buffer)
    }
    engine.stop()
}

do {
    try isolate(input: URL(fileURLWithPath: args[1]), output: URL(fileURLWithPath: args[2]), tail: tail)
} catch {
    fail("isolation failed: \(error)", 1)
}
