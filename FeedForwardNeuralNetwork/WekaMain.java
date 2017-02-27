import java.io.BufferedReader;
import java.io.FileReader;
import java.util.Random;
import java.util.Scanner;

import tucilweka.FFNN040;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Normalize;

public class WekaMain {
	
	public static void saveModel(String filename, Classifier c) throws Exception{
		//Save Model
		weka.core.SerializationHelper.write(filename, c);
	}
	
	public static Classifier loadModel(String filename) throws Exception{
		//Load Model
		return (Classifier) weka.core.SerializationHelper.read(filename);
	}
	
	public static void main(String args[]){
		long startTime = System.nanoTime();
		
		//Initialise
		BufferedReader br = null;
		BufferedReader bc = null;
		Classifier cls = null;
		Instances train = null;
		Instances data = null;
		Instances test = null;
		Scanner in = null;
		
		System.out.println("==== FFNN Trainer ====");
		in = new Scanner(System.in);
		System.out.println("1. Load model");
		System.out.println("2. Build model");
		int main = in.nextInt();
		in.nextLine();
		
		if(main==1){
			System.out.println("Enter filename");
			String name = in.nextLine();
			try {
				//Input test file
				bc = new BufferedReader(new FileReader("/home/chromeuser/Documents/TucilITB/ai/weka-3-8-0/data/Team_test.arff"));
				test = new Instances(bc);
				cls = loadModel(name);
				//Input training file
				br = new BufferedReader(new FileReader("/home/chromeuser/Documents/TucilITB/ai/weka-3-8-0/data/Team.arff"));
				data = new Instances(br); 
				
				//Input Options for Filter
				String[] options = new String[4];
				options[0] = "-S";                                    // "range"
				options[1] = "1.0";                                     // first attribute
				options[2] = "-T";
				options[3] = "0.0";
				Normalize norm = new Normalize();                         // new instance of filter
				norm.setOptions(options);                           // set options
				norm.setInputFormat(data);                          // inform filter about dataset **AFTER** setting options
				train = Filter.useFilter(data, norm);   // apply filter
				test = Filter.useFilter(test, norm);
				train.setClassIndex(train.numAttributes()-1);
				test.setClassIndex(train.numAttributes()-1);
			} catch (Exception e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
		} else if(main==2){
			try {
				//Input test file
				bc = new BufferedReader(new FileReader("/home/chromeuser/Documents/TucilITB/ai/weka-3-8-0/data/Team_test.arff"));
				test = new Instances(bc);
				//Input training file
				br = new BufferedReader(new FileReader("/home/chromeuser/Documents/TucilITB/ai/weka-3-8-0/data/Team.arff"));
				data = new Instances(br); 
				
				//Input Options for Filter
				String[] options = new String[4];
				options[0] = "-S";                                    // "range"
				options[1] = "1.0";                                     // first attribute
				options[2] = "-T";
				options[3] = "0.0";
				Normalize norm = new Normalize();                         // new instance of filter
				norm.setOptions(options);                           // set options
				norm.setInputFormat(data);                          // inform filter about dataset **AFTER** setting options
				train = Filter.useFilter(data, norm);   // apply filter
				test = Filter.useFilter(test, norm);
				test.setClassIndex(train.numAttributes()-1);
				//Option variables
				String seed = "0";
				String learnrate;
				String epoch;
				String isMultiLayer;
				String n_hidden;
				
				//Input Options for Classifier
				System.out.println("Input Epoch :");
				epoch = in.nextLine();
				System.out.println("Input Learn Rate :");
				learnrate = in.nextLine();
				System.out.println("Input Number of hidden layer (0/1) :");
				isMultiLayer = in.nextLine();
				if(isMultiLayer.equals("1")){
					System.out.println("Input Number of hidden nodes :");
					n_hidden = in.nextLine();
				} else {
					n_hidden = "0";
				}
				
				String[] cls_options = new String[10];
				//-S seed, -L learn rate, -M isMultiLayer, -E epoch, -H hidden node
				cls_options[0] = "-S";      
				cls_options[1] = seed;                
				cls_options[2] = "-L";
				cls_options[3] = learnrate;
				cls_options[4] = "-M";      
				cls_options[5] = isMultiLayer;                
				cls_options[6] = "-E";
				cls_options[7] = epoch;
				cls_options[8] = "-H";      
				cls_options[9] = n_hidden; 
				
				//Build Model
				train.setClassIndex(train.numAttributes()-1);	
				cls = new FFNN040();
				((FFNN040) cls).setOptions(cls_options);
				System.out.println("start build");
				cls.buildClassifier(train);
			}catch (Exception e){
				e.printStackTrace();
			}
		}
		try {
			//Evaluate Model
			Evaluation eval = new Evaluation(test);
			System.out.println("");
			System.out.println("1. Full Training");
			System.out.println("2. 10-fold cross validation");
			System.out.println("Choose evaluation method :");
			int opt = in.nextInt();
			if(opt==1){
				eval.evaluateModel(cls, test);
			} else if(opt==2){
				eval.crossValidateModel(cls, test, 10, new Random(0));
			} else {
				System.out.println("invalid option");
			}
			in.nextLine();
			System.out.println();
			System.out.println(eval.toSummaryString("\nClassifier Loaded\n", true));
			System.out.println(eval.toClassDetailsString());
			System.out.println(eval.toMatrixString());	
			
			long endTime = System.nanoTime();
			System.out.println("Took "+(endTime - startTime)/1000000 + " ms");
			
			System.out.println("Do you want to save your model? (y/n)");
			String save = in.nextLine();
			if(save.equals("y")){
				//Save model
				saveModel("FFNN.model", cls);
			}
			in.close();
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		
	}
}
