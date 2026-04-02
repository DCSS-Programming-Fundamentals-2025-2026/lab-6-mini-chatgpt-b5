using LabMiniChatGPT_A7;
using LabMiniChatGPT_A7.Configuration;
using LabMiniChatGPT_A7.FakeMathOps;
using LabMiniChatGPT_A7.Layers;
using LabMiniChatGPT_A7.State;

namespace A7.Tests;

public class Tests
{

    [Test]
    public void Project_CalculatesCorrectLogits()
    {
        var config = new TinyNNConfig (2,2,2);
        var weights = new TinyNNWeights(config);

        float[] hidden = { 1.0f, 2.0f };
        weights.OutputBias = new[] { 0.1f, 0.2f };

        weights.OutputWeights[0] = new[] { 0.5f, 0.1f };
        weights.OutputWeights[1] = new[] { 0.2f, 0.3f };
        
        float[] logits = LinearHead.Project(hidden, weights, config);

        //Logits[0] = (1.0 * 0.5) + (2.0*0.2) + 0.1 = 0.5+0.4+0.1 = 1.0f
        //Logits[1] = (1.0 * 0.1) + (2.0 * 0.3) + 0.2 = 0.1 + 0.6 + 0.2 = 0.9f
        Assert.Multiple(() =>
        {
            Assert.That(logits.Length, Is.EqualTo(2));
            Assert.That(logits[0], Is.EqualTo(1.0f).Within(0.00001)); //Within is need to correct loss because of flaot
            Assert.That(logits[1], Is.EqualTo(0.9f).Within(0.00001));
        });
    }

    [Test]
    public void Project_CalculatesCorrectWeights()
    {
        var config = new TinyNNConfig(2,2,2);
        var weights = new TinyNNWeights(config);
        
        float[] hidden = { 1.0f, 2.0f };
        weights.OutputBias = new[] { 0.1f, 0.2f };
        weights.OutputWeights[0] = new[] { 0.5f, 0.1f };
        weights.OutputWeights[1] = new[] { 0.2f, 0.3f };
        
        float initialBias0 = weights.OutputBias[0];
        float initialWeight00 = weights.OutputWeights[0][0];
        
        float[] dLogits = { 0.5f, -0.2f };
        float lr = 0.1f;
        
        float[] dHidden = LinearHead.Backward(dLogits, hidden, weights, config, lr);
        
        Assert.Multiple(() =>
        {
            Assert.That(dHidden, Is.Not.Null);
            Assert.That(dHidden.Length, Is.EqualTo(config.EmbeddingSize));
            Assert.That(weights.OutputBias[0], Is.Not.EqualTo(initialBias0));
            Assert.That(weights.OutputWeights[0][0], Is.Not.EqualTo(initialWeight00));
        });
    }
    
    [Test]
    public void NextTokenScores_ContextLongerThanLimit_SlicesCorrectly()
    {
        var config = new TinyNNConfig(10,  4, 2); 
        var weights = new TinyNNWeights(config);
        var fakeMath = new FakeMathOps();
        var model = new TinyNNModel(config, weights, fakeMath);

        int[] longContext = { 1, 2, 3, 4, 5 }; 
        int[] shortTail = { 4, 5 };            
        
        var longLogits = model.NextTokenScores(longContext);
        var shortLogits = model.NextTokenScores(shortTail);

        
        Assert.That(longLogits, Is.EqualTo(shortLogits));
    }
    
    [Test]
    public void NextTokenScores_EmptyContext_DoesNotCrash()
    {
        var config = new TinyNNConfig(10, 4, 3);
        var weights = new TinyNNWeights(config);
        var fakeMath = new FakeMathOps();
        var model = new TinyNNModel(config, weights, fakeMath);

        Assert.DoesNotThrow(() => model.NextTokenScores(Array.Empty<int>()));
    }
}