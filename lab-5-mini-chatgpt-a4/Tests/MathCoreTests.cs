using Lib.MathCore;
using NUnit.Framework; 
namespace Lib.MathCore.Tests;


[TestFixture] 
public class MathCoreTests
{
    
    [Test]
    public void Test_ArgMax_FindsMaximum()
    {
        
        float[] scores = { 0.5f, 1.2f, 5.0f, 0.8f }; 
 
        int result = MathOps.Default.ArgMax(scores);

        Assert.That(result, Is.EqualTo(2)); 
    }

    [Test]
    public void Test_Softmax_SumIsOne()
    {
 
        float[] logits = { 1.0f, 2.0f, 3.0f };

        float[] probs = MathOps.Default.Softmax(logits);

        float sum = 0;
        foreach (var p in probs) sum += p;

        Assert.That(sum, Is.EqualTo(1.0f).Within(0.0001f));
    }

    [Test]
    public void Test_Loss_CorrectPrediction()
    {

        float[] logits = { 20.0f, 0.0f, 0.0f }; 
        int target = 0; 

        float loss = MathOps.Default.CrossEntropyLoss(logits, target);

        Assert.That(loss, Is.LessThan(0.001f));
    }

    [Test]
    public void Test_Sample_ReturnsValidIndex()
    {
        float[] probs = { 0.1f, 0.7f, 0.2f };
        Random rng = new Random();

        for (int i = 0; i < 100; i++)
        {
            int result = MathOps.Default.SampleFromProbs(probs, rng);

            Assert.That(result, Is.GreaterThanOrEqualTo(0));
            Assert.That(result, Is.LessThan(probs.Length));
        }
    }
}