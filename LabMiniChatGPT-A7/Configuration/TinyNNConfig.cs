namespace LabMiniChatGPT_A7.Configuration;

public record TinyNNConfig(
    int VocabSize,
    int EmbeddingSize = 32,
    int ContextSize = 8
);
