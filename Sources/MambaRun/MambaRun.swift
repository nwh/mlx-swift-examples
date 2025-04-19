// TODO remove this temp file

import Foundation
import MLXLLM

let config = #"""
    {
        "architectures": [
            "MambaForCausalLM"
        ],
        "bos_token_id": 0,
        "conv_kernel": 4,
        "d_inner": 1536,
        "d_model": 768,
        "eos_token_id": 0,
        "expand": 2,
        "fused_add_norm": true,
        "hidden_act": "silu",
        "hidden_size": 768,
        "initializer_range": 0.1,
        "intermediate_size": 1536,
        "layer_norm_epsilon": 1e-05,
        "model_type": "mamba",
        "n_layer": 24,
        "num_hidden_layers": 24,
        "pad_token_id": 0,
        "pad_vocab_size_multiple": 8,
        "rescale_prenorm_residual": false,
        "residual_in_fp32": true,
        "rms_norm": true,
        "ssm_cfg": {},
        "state_size": 16,
        "time_step_floor": 0.0001,
        "time_step_init_scheme": "random",
        "time_step_max": 0.1,
        "time_step_min": 0.001,
        "time_step_rank": 48,
        "time_step_scale": 1.0,
        "torch_dtype": "float32",
        "transformers_version": "4.39.0.dev0",
        "use_bias": false,
        "use_cache": true,
        "use_conv_bias": true,
        "vocab_size": 50280
    }
    """#

@main
struct MambaRunMain {
    static func main() throws {
        print("Hello, world")
        let modelConfig = try JSONDecoder().decode(
            MambaConfiguration.self,
            from: config.data(using: .utf8)!
        )
        print(modelConfig)
    }
}
