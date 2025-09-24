// Copyright © 2025 Apple Inc.

import Foundation
import MLX
import MLXFast
import MLXLMCommon
import MLXNN

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

private class MixerNorm: Module, UnaryLayer {
    let eps: Float

    public init(eps: Float = 1e-5) {
        self.eps = eps
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        return
            MLXFast
            .rmsNorm(x, weight: MLX.ones([x.dim(-1)], dtype: x.dtype), eps: self.eps)
    }
}

private class MambaBlock: Module {

    let args: MambaConfiguration
    @ModuleInfo(key: "mixer_norm") var mixerNorm: MixerNorm?
    @ModuleInfo(key: "in_proj") var inProj: Linear
    @ModuleInfo(key: "conv1d") var conv1d: Conv1d

    public init(_ args: MambaConfiguration) {
        self.args = args
        if args.useBcdtRms {
            self._mixerNorm.wrappedValue = MixerNorm(eps: args.mixerRmsEps)
        }
        self._inProj.wrappedValue = Linear(
            args.hiddenSize, args.intermediateSize * 2, bias: args.useBias)
        self._conv1d.wrappedValue = Conv1d(
            inputChannels: args.intermediateSize,
            outputChannels: args.intermediateSize,
            kernelSize: args.convKernel,
            stride: 1,  //? TODO
            padding: 0,
            dilation: 0,  //? TODO
            groups: args.intermediateSize,
            bias: args.useConvBias
        )
    }
}
