// Copyright © 2025 Apple Inc.

import Foundation
import MLX
import MLXFast
import MLXLMCommon
import MLXNN

// TODO remove personal notes
// https://swiftpackageindex.com/ml-explore/mlx-swift-examples/main/documentation/mlxlmcommon/porting
// Properties that exist in weight files:
//  - @ModuleInfo(key: "layer_name") var layerName: Module
//  - @ParameterInfo var weights: MLXArray
//  - @ParameterInfo(key: "big_array") var bigArray: MLXArray
//
// For computed arrays that don't exist in weight files:
//  - private var _privateArray: MLXArray

// port of https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/models/mamba.py

struct StringKey: CodingKey, ExpressibleByStringLiteral {
    var intValue: Int? = nil
    var stringValue: String
    init?(intValue: Int) { return nil }
    init?(stringValue: String) { self.stringValue = stringValue }
    init(stringLiteral: StringLiteralType) {
        self.stringValue = stringLiteral
    }
}

public struct MambaConfiguration: Codable, Sendable {
    var modelType: String
    var vocabSize: Int
    var hiddenSize: Int
    var intermediateSize: Int
    var stateSize: Int
    var numHiddenLayers: Int
    var convKernel: Int
    var useBias: Bool
    var useConvBias: Bool
    var timeStepRank: Int
    var tieWordEmbeddings: Bool
    var useBcdtRms: Bool
    var mixerRmsEps: Float

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case vocabSize = "vocab_size"
        case hiddenSize = "hidden_size"
        case intermediateSize = "intermediate_size"
        case stateSize = "state_size"
        case numHiddenLayers = "num_hidden_layers"
        case convKernel = "conv_kernel"
        case useBias = "use_bias"
        case useConvBias = "use_conv_bias"
        case timeStepRank = "time_step_rank"
        case tieWordEmbeddings = "tie_word_embeddings"
        case useBcdtRms = "use_bcdt_rms"
        case mixerRmsEps = "mixer_rms_eps"
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        let fallback = try decoder.container(keyedBy: StringKey.self)

        modelType = try container.decode(String.self, forKey: .modelType)
        vocabSize = try container.decode(Int.self, forKey: .vocabSize)
        hiddenSize =
            try container
            .decodeIfPresent(Int.self, forKey: .hiddenSize)
            ?? fallback
            .decode(Int.self, forKey: "d_model")
        intermediateSize =
            try container
            .decodeIfPresent(Int.self, forKey: .intermediateSize)
            ?? fallback
            .decode(Int.self, forKey: "d_inner")
        stateSize =
            try container
            .decodeIfPresent(Int.self, forKey: .stateSize)
            ?? fallback
            .decode(Int.self, forKey: "d_state")
        numHiddenLayers =
            try container
            .decodeIfPresent(Int.self, forKey: .numHiddenLayers)
            ?? fallback
            .decodeIfPresent(Int.self, forKey: "n_layer")
            ?? fallback
            .decode(Int.self, forKey: "n_layers")
        convKernel =
            try container
            .decodeIfPresent(Int.self, forKey: .convKernel)
            ?? fallback
            .decode(Int.self, forKey: "d_conv")
        useBias =
            try container
            .decodeIfPresent(Bool.self, forKey: .useBias)
            ?? fallback
            .decode(Bool.self, forKey: "bias")
        useConvBias =
            try container
            .decodeIfPresent(Bool.self, forKey: .useConvBias)
            ?? fallback
            .decode(Bool.self, forKey: "conv_bias")

        if let timeStepRankAuto = try? container.decode(String.self, forKey: .timeStepRank),
            timeStepRankAuto == "auto"
        {
            timeStepRank = (hiddenSize + 15) / 16
        } else {
            timeStepRank = try container.decode(Int.self, forKey: .timeStepRank)
        }

        tieWordEmbeddings =
            try container.decodeIfPresent(Bool.self, forKey: .tieWordEmbeddings) ?? true
        useBcdtRms = try container.decodeIfPresent(Bool.self, forKey: .useBcdtRms) ?? false
        mixerRmsEps = try container.decodeIfPresent(Float.self, forKey: .mixerRmsEps) ?? 1e-6

        if modelType == "falcon_mamba" {
            useBcdtRms = true
        }
    }
}

private class MambaBlock: Module {

    let args: MambaConfiguration

    var _mixerNorm: ((MLXArray) -> MLXArray)? = nil

    @ModuleInfo(key: "in_proj") var inProj: Linear
    @ModuleInfo(key: "conv1d") var conv1d: Conv1d
    @ModuleInfo(key: "x_proj") var xProj: Linear
    @ModuleInfo(key: "dt_proj") var dtProj: Linear

    @ParameterInfo(key: "A_log") var aLog: MLXArray
    @ParameterInfo(key: "D") var d: MLXArray

    @ModuleInfo(key: "out_proj") var outProj: Linear

    public init(_ args: MambaConfiguration) {
        self.args = args
        if args.useBcdtRms {
            self._mixerNorm = {
                MLXFast.rmsNorm(
                    $0,
                    weight: MLX.ones([$0.dim(-1)], dtype: $0.dtype),
                    eps: args.mixerRmsEps)
            }
        }

        self._inProj.wrappedValue = Linear(
            args.hiddenSize, args.intermediateSize * 2, bias: args.useBias)

        self._conv1d.wrappedValue = Conv1d(
            inputChannels: args.intermediateSize,
            outputChannels: args.intermediateSize,
            kernelSize: args.convKernel,
            padding: 0,
            groups: args.intermediateSize,
            bias: args.useConvBias
        )

        self._xProj.wrappedValue = Linear(
            args.intermediateSize,
            args.timeStepRank + 2 * args.stateSize,
            bias: false
        )

        self._dtProj.wrappedValue = Linear(
            args.timeStepRank, args.intermediateSize, bias: true)

        // TODO A_log and D

        self._outProj = Linear(
            args.intermediateSize, args.hiddenSize, bias: args.useBias)
    }
}
