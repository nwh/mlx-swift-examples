// Copyright © 2025 Apple Inc.

import Foundation
import MLXLMCommon

// port of https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/models/mamba.py

// TODO support alternative JSON keys like d_model for hidden_size

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
    private var _useBcdtRms: Bool?
    public var useBcdtRms: Bool { _useBcdtRms ?? false }
    private let _mixerRmsEps: Float?
    public var mixerRmsEps: Float { _mixerRmsEps ?? 1e-6 }

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
        case _tieWordEmbeddings = "tie_word_embeddings"
        case _useBcdtRms = "use_bcdt_rms"
        case _mixerRmsEps = "mixer_rms_eps"
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)

        modelType = try container.decode(String.self, forKey: .modelType)
        vocabSize = try container.decode(Int.self, forKey: .vocabSize)
        hiddenSize = try container.decode(Int.self, forKey: .hiddenSize)
        intermediateSize = try container.decode(Int.self, forKey: .intermediateSize)
        stateSize = try container.decode(Int.self, forKey: .stateSize)
        numHiddenLayers = try container.decode(Int.self, forKey: .numHiddenLayers)
        convKernel = try container.decode(Int.self, forKey: .convKernel)
        useBias = try container.decode(Bool.self, forKey: .useBias)
        useConvBias = try container.decode(Bool.self, forKey: .useConvBias)
        if let intValue = try? container.decode(Int.self, forKey: .timeStepRank) {
            timeStepRank = intValue
        } else if let stringValue = try? container.decode(String.self, forKey: .timeStepRank),
            stringValue == "auto"
        {
            // timeStepRank = ceil(hiddenSize / 16)
            timeStepRank = (hiddenSize + 15) / 16
        } else {
            throw DecodingError.typeMismatch(
                String.self,
                .init(
                    codingPath: decoder.codingPath,
                    debugDescription: #"time_step_rank is not an Int or "auto""#))
        }
        _tieWordEmbeddings =
            try container
            .decodeIfPresent(Bool.self, forKey: ._tieWordEmbeddings)
        _useBcdtRms = try container.decodeIfPresent(Bool.self, forKey: ._useBcdtRms)
        if modelType == "falcon_mamba" {
            _useBcdtRms = true
        }
        _mixerRmsEps = try container.decodeIfPresent(Float.self, forKey: ._mixerRmsEps)
    }
}
