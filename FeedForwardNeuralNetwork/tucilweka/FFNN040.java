package tucilweka;

import java.util.Arrays;
import java.util.Enumeration;
import java.util.List;
import java.util.Random;

import weka.classifiers.AbstractClassifier;
import weka.core.Capabilities;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Option;
import weka.core.OptionHandler;
import weka.core.WeightedInstancesHandler;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.NominalToBinary;
import weka.filters.unsupervised.attribute.Normalize;
import weka.filters.unsupervised.attribute.Standardize;

import java.lang.Math;

public class FFNN040 extends AbstractClassifier implements OptionHandler, WeightedInstancesHandler {
	
	public Instances instas;
	public int clsIdx;
	
	//Option variables
	public int epoch = 10000;
	public double learnrate = 0.05;
	public int seed = 0;
	public boolean isMultiLayer = true;
	
	//Nilai dari tiap-tiap node pada layer
	public double[] input_layer;
	public double[] output_layer;
	public double[] hidden_layer;
	
	//Filters
	public Normalize norm;
	public Standardize std;
	public NominalToBinary ntb;
	
	//Nilai bobot antar layer. Misalkan input berjumlah 2 node, hidden 2 node, dan output 2 node, maka
	//node dinomori input 0,1; hidden 2,3; output 4,5. Bobot antara node 0 dengan node 2 diisi di weight_matrix[2][0],
	//node 1 dengan node 2 di weight_matrix[2][1], (yang lebih besar di kiri) dst.
	public double[][] weight_matrix;
	
	//Jumlah node tiap layer, diisi saat inisialisasi
	public int n_input;
	public int n_output;
	public int n_hidden;
	
	public void printMatrix(){
		int countHidden=0;
		int countOutput=0;
		for(int hidden=n_input;hidden<n_input+n_hidden;hidden++){
	    	for(int input=0; input<n_input;input++){
	    		
	    		System.out.println("Hidden element "+hidden+input+" "+weight_matrix[hidden][input]);
	    		countHidden++;
		    }
	    }
		System.out.println(countHidden);
	    for(int output=n_input+n_hidden;output<n_input+n_hidden+n_output;output++){
	    	for(int hidden=n_input;hidden<n_input+n_hidden;hidden++){
	    		System.out.println("Output element "+output+hidden+" "+weight_matrix[output][hidden]);
	    		countOutput++;
		    }
	    }
	    System.out.println(countOutput);
		
	}
	
	@Override
	public double classifyInstance(Instance arg0) throws Exception {
		// TODO Auto-generated method stub
		ntb.input(arg0);
		Instance newarg0 = ntb.output();
		norm.input(newarg0);
		newarg0 = norm.output();
		
		calculateOutput(newarg0);
		double max = -1;
		for(int i=0; i<n_output;++i){
			if(output_layer[i]>max){
				max = output_layer[i];
			}
		}
		return max;
	}
	
	public double[] distributionForInstance(Instance arg0) throws Exception{
		ntb.input(arg0);
		Instance newarg0 = ntb.output();
		norm.input(newarg0);
		newarg0 = norm.output();
		
		calculateOutput(newarg0);
		return output_layer;
	}

	@Override
	public String[] getOptions() {
		return super.getOptions();
	}

	@Override
	public Enumeration<Option> listOptions() {
		return super.listOptions();
	}

	@Override
	public void setOptions(String[] options) throws Exception {
		List<String> listopt = Arrays.asList(options);
		if(listopt.contains("-S")){
			seed = Integer.parseInt(listopt.get(listopt.indexOf("-S")+1));
		}
		if(listopt.contains("-M")){
			if(listopt.get(listopt.indexOf("-M")+1).equals("1")){
				isMultiLayer = true;
			} else {
				isMultiLayer = false;
			}
		}
		if(listopt.contains("-E")){
			epoch = Integer.parseInt(listopt.get(listopt.indexOf("-E")+1));
		}
		if(listopt.contains("-L")){
			learnrate = Double.parseDouble(listopt.get(listopt.indexOf("-L")+1));
		}
		if(listopt.contains("-H")){
			n_hidden = Integer.parseInt(listopt.get(listopt.indexOf("-H")+1));
		}
	}

	@Override
	public String toString() {
		return super.toString();
	}
	
	public void initWeightMatrix(double[][] weight_matrix){
		Random rand=  new Random(seed);
		if(isMultiLayer){
			for(int hidden=n_input;hidden<n_input+n_hidden;hidden++){
		    	for(int input=0; input<n_input;input++){
		    		weight_matrix[hidden][input] = ((rand.nextDouble()-0.5)*2)/Math.sqrt(n_input);
			    }
		    }
		    for(int output=n_input+n_hidden;output<n_input+n_hidden+n_output;output++){
		    	for(int hidden=n_input;hidden<n_input+n_hidden;hidden++){
			    	weight_matrix[output][hidden] = ((rand.nextDouble()-0.5)*2)/Math.sqrt(n_input);
			    }
		    	weight_matrix[n_output+n_input+n_hidden][output] = ((rand.nextDouble()-0.5)*2)/Math.sqrt(n_input);
		    }
		} else {
			for(int output=n_input;output<n_input+n_output;output++){
				for(int input=0; input<n_input;input++){
		    		weight_matrix[output][input] = ((rand.nextDouble()-0.5)*2)/Math.sqrt(n_input);
			    }
		    }
		}
	}
	
	@Override
	public void buildClassifier(Instances instances) throws Exception {
		if (instances.classIndex() < 0) {
			throw new Exception ("No class attribute assigned to instances");
		}
		
		// Hilangkan data tanpa kelas
	    instas = new Instances(instances, 0, instances.numInstances());
	    instas.deleteWithMissingClass();
	    
	    //Input Options for Filter
		String[] options = new String[4];
		options[0] = "-S";                                    // "range"
		options[1] = "1.0";                                     // first attribute
		options[2] = "-T";
		options[3] = "0.0";
		
		String[] opti = new String[2];
        opti[0] = "-R";                                    // "range"
		opti[1] = "first-26,28-last"; 
		
		norm = new Normalize();
		ntb = new NominalToBinary();
		std = new Standardize();
		norm.setOptions(options);
		ntb.setOptions(opti);
		
		ntb.setInputFormat(instances);
		
		
		instas = Filter.useFilter(instas, ntb);
		
		norm.setInputFormat(instas);
		instas = Filter.useFilter(instas, norm);
		
		
	    //Inisialisasi struktur data
	    clsIdx = 39;
	    instas.setClassIndex(clsIdx);
	    n_input = instas.numAttributes(); //+1 karena bias node
	    n_output = instas.numClasses();
	    input_layer = new double[n_input];
	    output_layer = new double[n_output];
	    hidden_layer = new double[n_hidden];
	    
	    int total = n_input+n_output+n_hidden;
	    //Inisialisasi weight dengan 1
	    weight_matrix = new double[total+1][total+1];
	    initWeightMatrix(weight_matrix);
	    
	    int j=0;
	    //Enumerasi instance dan panggil calculateOutput()
	   while(j<epoch){ 
	    for(int i =0;i< instas.size();i++){
	    	Instance insta = instas.get(i);
	    	calculateOutput(insta);
	    	backPropagate(insta,learnrate);
	
	    }
            if (j % (epoch / 10) == 0) {
                //System.out.println("epoch "+j);
            }
	    
	    j++;
	   }
	   
	}
	
	@Override
	public Capabilities getCapabilities(){
		//disable all
		//only enable Numeric data
		return super.getCapabilities();
	}
	
	//Lakukan perhitungan terhadap 1 instance yang didapat dari buildClassifier. Simpan hasil perhitungan di output_layer
	public void calculateOutput(Instance instance){
		int i;
        int j;
        double sigma;
        double esigma;

        //Assign nilai Input
        int c = 0;
        for (i = 0; i < n_input - 1; i++){
        	if(i==clsIdx){
        		c = 1;
        	}
            input_layer[i] = instance.value(i+c);
        }

        //Bias Input
        input_layer[n_input - 1] = 1;
        
        if(isMultiLayer){
	        //Calculate Output for Hidden Layer
			for (i = n_input; i < n_input + n_hidden; i++){
	            sigma = 0;
	            esigma = 0;
	            for (j = 0; j < n_input; j++){
	                sigma = sigma + input_layer[j] * weight_matrix[i][j];
	            }
	            esigma = Math.exp(-sigma);
	            hidden_layer[i-n_input] = 1/(1+esigma);
	        }
	
	        //Calculate Output for Output Layer
			for (i = n_hidden + n_input; i < n_hidden + n_output + n_input; i++){
	            sigma = 0;
	            esigma = 0;
	            for (j = n_input; j < n_input + n_hidden; j++){
	                sigma = sigma + hidden_layer[j-n_input] * weight_matrix[i][j];
	            }
	            
	            //Calculate bias
	            sigma = sigma + weight_matrix[n_output+n_input+n_hidden][i];
	            
	            esigma = Math.exp(-sigma);
	            output_layer[i-n_input-n_hidden] = 1/(1+esigma);
	        }
        } else {
        	for (i = n_input; i < n_output + n_input; i++){
	            sigma = 0;
	            esigma = 0;
	            for (j = 0; j < n_input; j++){
	                sigma = sigma + input_layer[j] * weight_matrix[i][j];
	            }
	            esigma = Math.exp(-sigma);
	            output_layer[i-n_input] = 1/(1+esigma);
	        }
        }
	}
	
	//Mengupdate weight_matrix dengan metode backpropagation
	public void backPropagate(Instance instance, double learnrate){
		if(isMultiLayer){
			//New Weight = old weight + learningrate*Error_out1*input (input is output from hidden layer)
			//Error_out1 = output1*(1-output1)*(target - output1)
			
			//Calculate error in output layer
			double[] err_output = new double[n_output];
			for(int i=0; i<n_output; ++i){
				double index_target = instance.classValue();
				double output = output_layer[i];
				double target = 0;
				if(i==index_target){
					target = 1;
				}
				err_output[i] = output*(1-output)*(target - output);
			}
			
			//Calculate error in hidden layer
			double[] err_hidden = new double[n_hidden];
			for(int i=0; i<n_hidden; ++i){
				double propagate = 0;
				
				//calculate propagation
				for(int j=0;j<n_output;++j){
					propagate = propagate + (err_output[j]*weight_matrix[j+n_input+n_hidden][i+n_input]);
				}
				
				double output = hidden_layer[i];
				err_hidden[i] = output*(1-output)*(propagate);
			}
			
			
			//Update all weight in the end
			for(int hidden=n_input;hidden<n_input+n_hidden;hidden++){
		    	for(int input=0; input<n_input;input++){
		    		weight_matrix[hidden][input] = weight_matrix[hidden][input] + learnrate*(err_hidden[hidden-n_input])*input_layer[input];
			    }
		    }
		    for(int output=n_input+n_hidden;output<n_input+n_hidden+n_output;output++){
		    	for(int hidden=n_input;hidden<n_input+n_hidden;hidden++){
			    	weight_matrix[output][hidden] = weight_matrix[output][hidden] + learnrate*(err_output[output-n_input-n_hidden])*hidden_layer[hidden-n_input];
			    }
		    	//Update bias weights
		    	weight_matrix[n_output+n_input+n_hidden][output] = weight_matrix[n_output+n_input+n_hidden][output] + learnrate*(err_output[output-n_input-n_hidden]);
		    }
		    
		} else {
			//Single layer perceptron
			//Calculate error in output layer
			double[] err_output = new double[n_output];
			for(int i=0; i<n_output; ++i){
				double index_target = instance.classValue();
				double output = output_layer[i];
				double target = 0;
				if(i==index_target){
					target = 1;
				}
				err_output[i] = output*(1-output)*(target - output);
			}
			
			//Update all weight in the end
		    for(int output=n_input;output<n_input+n_output;output++){
		    	for(int input=0;input<n_input;input++){
			    	weight_matrix[output][input] = weight_matrix[output][input] + learnrate*(err_output[output-n_input])*input_layer[input];
			    }
		    }
		}
	}
}
