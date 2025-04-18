// Copyright © 2025 Apple Inc.

import Foundation

// port of https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/models/mamba.py

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
    private let _tieWordEmbeddings: Bool?
    public var tieWordEmbeddings: Bool { _tieWordEmbeddings ?? true }
    private let _useBcdtRms: Bool?
    public var useBcdtRms: Bool { _useBcdtRms ?? false }
    private let _mixerRmsEps: Float?
    public var mixerRmsEps: Float { _mixerRmsEps ?? 1e-6 }

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case vocabSize = "vocal_size"
        case hiddenSize = "hidden_size"
        case intermediateSize = "intermediate_size"
        case stateSize = "state_size"
        case numHiddenLayers = "num_hidden_layers"
        case convKernel = "conv_kernel"
        case useBias = "use_bias"
        case useConvBias = "use_conv_bias"
        case timeStepRank = "time_step_rank"
        case _tieWordEmbeddings = "tie_word_embeddings"
        case _useBcdtRms = "use_bcdt_rms"
        case _mixerRmsEps = "mixer_rms_eps"
    }
}
