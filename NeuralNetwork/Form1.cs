using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;

namespace NeuralNetwork
{
    public partial class Form1 : Form
    {
        // 16 test sets with 4 inputs each, -1 represents a black square, 1 represents a white square
        int[,] inputs = new int[16, 4] {
                                        { -1, -1, -1, -1 },
                                        { -1, -1, -1, 1 },
                                        { -1, -1, 1, -1 },
                                        { -1, -1, 1, 1 },
                                        { -1, 1, -1, -1 },
                                        { -1, 1, -1, 1 },
                                        { -1, 1, 1, -1 },
                                        { -1, 1, 1, 1 },
                                        { 1, -1, -1, -1 },
                                        { 1, -1, -1, 1 },
                                        { 1, -1, 1, -1 },
                                        { 1, -1, 1, 1 },
                                        { 1, 1, -1, -1 },
                                        { 1, 1, -1, 1 },
                                        { 1, 1, 1, -1 },
                                        { 1, 1, 1, 1 }
                                    };
        // correct outputs in order of corresponding inputs, based on 2-4 white squares = bright, and 0-1 white squares = dark
        int[] answers = { 0, 0, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1 };

        List<double> outputResults = new List<double>(); // used for storing each output result for visual display in the ui
        List<double[]> neuronOutputs = new List<double[]>(); // same as above, but for individual neuron outputs

        public Form1()
        {
            InitializeComponent();
            // minor tweaks to the labels to make sure they dont become too large
            label5.AutoSize = false;
            label6.AutoSize = false;
            label7.AutoSize = false;
            label8.AutoSize = false;
            label9.AutoSize = false;

            label1.Text = "";
            label2.Text = "";
            label3.Text = "";
            label4.Text = "";
            label5.Text = "";
            label6.Text = "";
            label7.Text = "";
            label8.Text = "";
            label9.Text = "";

            textBox1.AppendText("--------------------Training Log--------------------- \r\n");

            compute();
        }

        // retrains the neural network with a call to compute() which is logged in the textbox element
        public void button1_Click(object sender, EventArgs e)
        {
            compute();
        }

        // used by the ui to display an animation of the neural network
        public async Task displayResultsAsync()
        {
            pictureBox5.Visible = true;
            double display = 0;
            double[] neuronOut = new double[4];
            for (int i = 0; i < 16; i++)
            {
                
                label1.Text = inputs[i, 0].ToString();
                label2.Text = inputs[i, 1].ToString();
                label3.Text = inputs[i, 2].ToString();
                label4.Text = inputs[i, 3].ToString();
                label5.Text = outputResults[outputResults.Count - 16 + i].ToString();

                neuronOut = neuronOutputs[neuronOutputs.Count - 16 + i];
                label6.Text = neuronOut[0].ToString();
                label7.Text = neuronOut[1].ToString();
                label8.Text = neuronOut[2].ToString();
                label9.Text = neuronOut[3].ToString();

                loadEnvironment(i);

                display = outputResults[outputResults.Count - 16 + i];
                display = Math.Round(display);
                pictureBox5.Image = (display == 1 ? sun.Image : moon.Image);
                await Task.Delay(3000);
            }
        }

        // main function of the neural network to train and store the outputs using backprop & feedforward
        public void compute()
        {
            int epoch = 0; // used for iterating through tests N times

            // hidden layer of neurons
            Neuron hidden1 = new Neuron();
            Neuron hidden2 = new Neuron();
            Neuron hidden3 = new Neuron();
            Neuron hidden4 = new Neuron();

            //output neuron
            Neuron output = new Neuron();
            output.randomizeWeights();

            hidden1.randomizeWeights();
            hidden2.randomizeWeights();
            hidden3.randomizeWeights();
            hidden4.randomizeWeights();

            //double errorRate = 1;

            //output.error = 1;
            while (epoch < 100) // iterate based on the output error OR use epochs > 100 for reliable numbers
            {
                //errorRate = 0;
                epoch++;
                for (int i = 0; i < 16; i++)
                {
                    // set each neurons inputs to the training set
                    hidden1.inputs = new double[] { inputs[i, 0], inputs[i, 1], inputs[i, 2], inputs[i, 3] };
                    hidden2.inputs = new double[] { inputs[i, 0], inputs[i, 1], inputs[i, 2], inputs[i, 3] };
                    hidden3.inputs = new double[] { inputs[i, 0], inputs[i, 1], inputs[i, 2], inputs[i, 3] };
                    hidden4.inputs = new double[] { inputs[i, 0], inputs[i, 1], inputs[i, 2], inputs[i, 3] };
                    
                    // set the outputnode inputs to the results of the 4 neurons sigmoid transfer
                    output.inputs = new double[] { hidden1.output(), hidden2.output(), hidden3.output(), hidden4.output() };

                    // display the training set and the "output of the output" neuron
                    // we set the output nodes inputs above, and now we're using those inputs * weights + bias formula in the output function
                    textBox1.AppendText(inputs[i, 0] + " " + inputs[i, 1] + " " + inputs[i, 2] + " " + inputs[i, 3] + " = " + output.output() + "\r\n");
                    outputResults.Add(output.output());
                    neuronOutputs.Add(new double[] { hidden1.output(), hidden2.output(), hidden3.output(), hidden4.output() });

                    // calculate the error
                    output.error = sigmoid.derivative(output.output()) * (answers[i] - output.output());
                    output.tweakWeights(); 

                    // calculate the neurons error rates based on their individual outputs * the error rate of the output neuron * the weights of the output neuron
                    // the sigmoid deriv is used for adjusting the weights in order to reduce the error rate of each neuron 
                    hidden1.error = sigmoid.derivative(hidden1.output()) * output.error * output.weights[0];
                    hidden2.error = sigmoid.derivative(hidden2.output()) * output.error * output.weights[1];
                    hidden3.error = sigmoid.derivative(hidden3.output()) * output.error * output.weights[2];
                    hidden4.error = sigmoid.derivative(hidden4.output()) * output.error * output.weights[3];

                    // the line below is irrelevant since our sigmoid function takes care of the error rate
                    //double error = answers[i] - output.output();

                    // tweak the neurons weights using the error rates by adding the error rate * input value to the weights
                    hidden1.tweakWeights();
                    hidden2.tweakWeights();
                    hidden3.tweakWeights();
                    hidden4.tweakWeights();

                    //errorRate += Math.Abs(error);
                }
            }
        }

        // used for iterating through environments in the UI
        public void loadEnvironment(int i)
        {
            int set;
            switch (i)
            {
                case 0:
                    pictureBox1.BackColor = Color.Black;
                    pictureBox2.BackColor = Color.Black;
                    pictureBox3.BackColor = Color.Black;
                    pictureBox4.BackColor = Color.Black;
                    set = 0;
                    break;

                case 1:
                    pictureBox1.BackColor = Color.Black;
                    pictureBox2.BackColor = Color.Black;
                    pictureBox3.BackColor = Color.Black;
                    pictureBox4.BackColor = Color.White;
                    set = 1;
                    break;

                case 2:
                    pictureBox1.BackColor = Color.Black;
                    pictureBox2.BackColor = Color.Black;
                    pictureBox3.BackColor = Color.White;
                    pictureBox4.BackColor = Color.Black;
                    set = 2;
                    break;

                case 3:
                    pictureBox1.BackColor = Color.Black;
                    pictureBox2.BackColor = Color.Black;
                    pictureBox3.BackColor = Color.White;
                    pictureBox4.BackColor = Color.White;
                    set = 3;
                    break;

                case 4:
                    pictureBox1.BackColor = Color.Black;
                    pictureBox2.BackColor = Color.White;
                    pictureBox3.BackColor = Color.Black;
                    pictureBox4.BackColor = Color.Black;
                    set = 4;
                    break;

                case 5:
                    pictureBox1.BackColor = Color.Black;
                    pictureBox2.BackColor = Color.White;
                    pictureBox3.BackColor = Color.Black;
                    pictureBox4.BackColor = Color.White;
                    set = 5;
                    break;

                case 6:
                    pictureBox1.BackColor = Color.Black;
                    pictureBox2.BackColor = Color.White;
                    pictureBox3.BackColor = Color.White;
                    pictureBox4.BackColor = Color.Black;
                    set = 6;
                    break;

                case 7:
                    pictureBox1.BackColor = Color.Black;
                    pictureBox2.BackColor = Color.White;
                    pictureBox3.BackColor = Color.White;
                    pictureBox4.BackColor = Color.White;
                    set = 7;
                    break;

                case 8:
                    pictureBox1.BackColor = Color.White;
                    pictureBox2.BackColor = Color.Black;
                    pictureBox3.BackColor = Color.Black;
                    pictureBox4.BackColor = Color.Black;
                    set = 8;
                    break;

                case 9:
                    pictureBox1.BackColor = Color.White;
                    pictureBox2.BackColor = Color.Black;
                    pictureBox3.BackColor = Color.Black;
                    pictureBox4.BackColor = Color.White;
                    set = 9;
                    break;

                case 10:
                    pictureBox1.BackColor = Color.White;
                    pictureBox2.BackColor = Color.Black;
                    pictureBox3.BackColor = Color.White;
                    pictureBox4.BackColor = Color.Black;
                    set = 10;
                    break;

                case 11:
                    pictureBox1.BackColor = Color.White;
                    pictureBox2.BackColor = Color.Black;
                    pictureBox3.BackColor = Color.White;
                    pictureBox4.BackColor = Color.White;
                    set = 11;
                    break;

                case 12:
                    pictureBox1.BackColor = Color.White;
                    pictureBox2.BackColor = Color.White;
                    pictureBox3.BackColor = Color.Black;
                    pictureBox4.BackColor = Color.Black;
                    set = 12;
                    break;

                case 13:
                    pictureBox1.BackColor = Color.White;
                    pictureBox2.BackColor = Color.White;
                    pictureBox3.BackColor = Color.Black;
                    pictureBox4.BackColor = Color.White;
                    set = 13;
                    break;

                case 14:
                    pictureBox1.BackColor = Color.White;
                    pictureBox2.BackColor = Color.White;
                    pictureBox3.BackColor = Color.White;
                    pictureBox4.BackColor = Color.Black;
                    set = 14;
                    break;

                case 15:
                    pictureBox1.BackColor = Color.White;
                    pictureBox2.BackColor = Color.White;
                    pictureBox3.BackColor = Color.White;
                    pictureBox4.BackColor = Color.White;
                    set = 15;
                    break;
            }
        }

        private void button2_Click(object sender, EventArgs e)
        {
            displayResultsAsync();
        }

        private void pictureBox_Click(object sender, EventArgs e)
        {

        }

        private void button3_Click(object sender, EventArgs e)
        {
            MessageBox.Show("Welcome!, this is an artificial neural network using back propagation and feed forward techniques. We have 4 squares, each of which represent " +
                            "an input. Black squares are represented as -1 and white squares are 1. If we have 2-4 white squares, the image is considered bright. " +
                            "If we have 3-4 black squares, the image is considered dark. Bright and dark are shown as a sun and moon as the final output! For more" +
                            "details, please check the readme file for an indepth explanation of how the algorithm functions!");
        }
    }
}
