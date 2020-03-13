using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetwork
{
    public class Neuron
    {
        public double[] inputs = new double[4];
        public double[] weights = new double[4];
        public double error;

        public double bias;

        Random r = new Random();

        // uses the sigmoid class to return the sigmoid of our (weights * inputs + ... + bias) formula
        public double output()
        {
            return sigmoid.output(weights[0] * inputs[0] + weights[1] * inputs[1] + weights[2] * inputs[2] + weights[3] * inputs[3] + bias);
        }

        // called to initially randomize the weights for each neuron, a number between 0 and 1 is chosen
        public void randomizeWeights()
        {
            //double min = -.5;
            //double max = .5;
            Random rand = new Random();

            for (int i = 0; i < 4; i++)
            {
                //weights[i] = rand.NextDouble() * (max - min) + min;
                weights[i] = rand.NextDouble();
            }
            //bias = rand.NextDouble() * (max - min) + min;
            bias = rand.NextDouble();
        }

        // adjusts weights by adding the error * input data
        // bias weight is simply the bias with the error value added
        public void tweakWeights()
        {
            for (int i = 0; i < 4; i++)
            {
                weights[i] += error * inputs[i];
            }
            bias += error;
        }
    }
}
