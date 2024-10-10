from transformers import PretrainedConfig

class DeCodonConfig(PretrainedConfig):
    def __init__(
        self,
        vocab_size=70,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=512,
        type_vocab_size=2,
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        pad_token_id=0,
        position_embedding_type="rotary",
        use_cache=True,
        classifier_dropout=None,
        gamma_init=1.0,
        use_rotary_emb=True,
        rotary_theta=1e4,
        use_flash_attn=False,
        **kwargs,
    ):
        super().__init__(
            pad_token_id=pad_token_id,
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            intermediate_size=intermediate_size,
            hidden_act=hidden_act,
            hidden_dropout_prob=hidden_dropout_prob,
            attention_probs_dropout_prob=attention_probs_dropout_prob,
            max_position_embeddings=max_position_embeddings,
            type_vocab_size=type_vocab_size,
            initializer_range=initializer_range,
            layer_norm_eps=layer_norm_eps,
            position_embedding_type=position_embedding_type,
            use_cache=use_cache,
            classifier_dropout=classifier_dropout,
            gamma_init=gamma_init,
            use_rotary_emb=use_rotary_emb,
            rotary_theta=rotary_theta,
            use_flash_attn=use_flash_attn,
            is_decoder=kwargs.pop("is_decoder", True),
            **kwargs,
        )


class DeCodonForSequenceTaskConfig(DeCodonConfig):
    def __init__(
        self,
        task_name="NoName",
        num_labels=2,
        num_tasks=1,
        loss_fn="huber",
        cls_num_hidden_layers=1,
        cls_hidden_size=128,
        cls_dropout_prob=0.1,
        cls_hidden_act="relu",
        cls_type="cls",
        cls_num_attention_heads=8,
        cls_use_rotary_emb=False,
        cls_rotary_theta=1e4,
        num_filters=128,
        kernel_size=3,
        stride=1,
        dilation=1,
        pooling_size=2,
        pooling_type="max",
        layer_indices=-1,
        reduction="mean",
        layer_reduction="none",
        problem_type="classification",
        **kwargs,
    ):

        if problem_type == "classification":
            problem_type_ = "single_label_classification"
        else:
            problem_type_ = problem_type

        super().__init__(
            task_name=task_name,
            num_labels=num_labels,
            num_tasks=num_tasks,
            loss_fn=loss_fn,
            cls_num_hidden_layers=cls_num_hidden_layers,
            cls_hidden_size=cls_hidden_size,
            cls_dropout_prob=cls_dropout_prob,
            cls_hidden_act=cls_hidden_act,
            cls_num_attention_heads=cls_num_attention_heads,
            cls_use_rotary_emb=cls_use_rotary_emb,
            cls_rotary_theta=cls_rotary_theta,
            cls_type=cls_type,
            num_filters=num_filters,
            kernel_size=kernel_size,
            stride=stride,
            dilation=dilation,
            pooling_size=pooling_size,
            pooling_type=pooling_type,
            layer_indices=layer_indices,
            reduction=reduction,
            layer_reduction=layer_reduction,
            problem_type=problem_type_,
            **kwargs,
        )

        self.problem_type = problem_type
