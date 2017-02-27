/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package tucilweka;

import java.io.BufferedReader;
import java.io.FileReader;
import java.util.Random;
import java.util.Scanner;

import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ConverterUtils;

import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.lazy.IBk;

import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Discretize;
import weka.filters.unsupervised.attribute.NominalToBinary;
import weka.filters.unsupervised.attribute.Normalize;
import weka.filters.unsupervised.attribute.Remove;
import weka.filters.unsupervised.attribute.Standardize;

public class TucilWeka {

	
	public static void saveModel(String filename, Classifier c) throws Exception{
		//Save Model
		weka.core.SerializationHelper.write(filename, c);
	}
	
	public static Classifier loadModel(String filename) throws Exception{
		//Load Model
		return (Classifier) weka.core.SerializationHelper.read(filename);
	}
        
    public static void main(String[] args) throws Exception {
        Scanner sc = new Scanner(System.in);
        int hahahaha = sc.nextInt();
        if (hahahaha == 0) {
            // Menentukan parameter penguji
            int folds = 10;
            int seed = 1;
            Random rand = new Random(seed);

            // Membaca dataset
            ConverterUtils.DataSource source = new ConverterUtils.DataSource("C:\\Users\\HP\\Desktop\\mush.arff");
            Instances dataset = source.getDataSet();
            dataset.setClassIndex(0); 

            source = new ConverterUtils.DataSource("C:\\Users\\HP\\Desktop\\mush_test.arff");
            Instances dotoset = source.getDataSet();
            dotoset.setClassIndex(0); 
            
            // Membuat classifier default (NaiveBayes)
            Classifier cls = new NaiveBayes();
            cls.buildClassifier(dataset);

            // Inisasi
            int i = 0;
            boolean discret = false;
            Discretize disc = new Discretize();

            System.out.println("WEKA, iris.arff loaded");
            System.out.println("Options:");
            System.out.println("1. Normal");
            System.out.println("2. Discretize");
            System.out.print("Pilihan: ");
            i = sc.nextInt();

            if (i == 2) {
                // Ubah data uji dengan discretize (numeric ke nominal)
                discret = true;
                disc.setInputFormat(dataset);
                dataset = Filter.useFilter(dataset, disc);
                disc.setInputFormat(dotoset);
                dotoset = Filter.useFilter(dotoset, disc);
            }
            i = 0;

            while (i != 2) {
                System.out.println("Options:");
                System.out.println("1. Pembelajaran");
                System.out.println("2. Exit");
                System.out.print("Pilihan: ");
                i = sc.nextInt();

                switch (i) {
                    case 1:
                        // Pembelajaran
                        int k = 0;
                        while (k != 8) {
                            System.out.println("Options:");
                            System.out.println("1. 10 Folds CV - NB");
                            System.out.println("2. 10 Folds CV - 1NN");
                            System.out.println("3. Full Training - NB");
                            System.out.println("4. Full Training - 1NN");
                            System.out.println("5. Read Model");
                            System.out.println("6. Save Model");
                            System.out.println("7. Test Instance");
                            System.out.println("8. Back");
                            System.out.print("Pilihan: ");
                            k = sc.nextInt();

                            Evaluation eval = new Evaluation(dataset);

                            switch (k) {
                                case 1:
                                    // 10 Folds CV - NaiveBayes
                                    System.out.println("10 FCV - NB");

                                    // Latih classifier dengan iris.arff (NaiveBayes)
                                    cls = new NB();
                                    cls.buildClassifier(dataset);
                                    System.out.println(cls);

                                    // Uji hasil classifier (10 FCV)
                                    eval.crossValidateModel(cls, dataset, folds, rand);
                                    System.out.println(eval.toSummaryString());
                                    System.out.println(eval.toClassDetailsString());
                                    System.out.println(eval.toMatrixString());
                                    break;
                                case 2:
                                    // 10 Folds CV - 1NN 
                                    System.out.println("10 FCV - 1NN");

                                    // Latih classifier dengan iris.arff (1NN)
                                    cls = new IBk();
                                    cls.buildClassifier(dataset);
                                    System.out.println(cls);

                                    // Uji hasil classifier (10 FCV)
                                    eval.crossValidateModel(cls, dataset, folds, rand);
                                    System.out.println(eval.toSummaryString());
                                    System.out.println(eval.toClassDetailsString());
                                    System.out.println(eval.toMatrixString());
                                    break;
                                case 3:
                                    // Full Training - NaiveBayes
                                    System.out.println("FT - NB");

                                    // Latih classifier dengan iris.arff (NaiveBayes)
                                    cls = new NB();
                                    cls.buildClassifier(dataset);
                                    System.out.println(cls);

                                    // Uji hasil classifier (Full Train)
                                    eval.evaluateModel(cls, dotoset);
                                    System.out.println(eval.toSummaryString());
                                    System.out.println(eval.toClassDetailsString());
                                    System.out.println(eval.toMatrixString());
                                    break;
                                case 4:
                                    // Full Training - 1NN
                                    System.out.println("FT - 1NN");

                                    // Latih classifier dengan iris.arff (1NN)
                                    cls = new NaiveBayes();
                                    cls.buildClassifier(dataset);
                                    System.out.println(cls);

                                    // Uji hasil classifier (Full Train)
                                    eval.evaluateModel(cls, dataset);
                                    System.out.println(eval.toSummaryString());
                                    System.out.println(eval.toClassDetailsString());
                                    System.out.println(eval.toMatrixString());
                                    break;
                                case 5:
                                    // Membaca model dari irisnew.model
                                    cls = (Classifier)weka.core.SerializationHelper.read("irisnew.model");
                                    System.out.println("Model irisnew.model successfully loaded!");
                                    break;
                                case 6:
                                    // Menuliskan model hasil pembelajaran
                                    weka.core.SerializationHelper.write("irisnew.model", cls);
                                    System.out.println("Model irisnew.model successfully saved!");
                                    break;
                                case 7:
                                    // Melakukan pengklasifikasian instance pengguna
                                    int natt = dataset.numAttributes();
                                    Instance inst = new DenseInstance(natt);

                                    // Membuat instance berdasarkan masukkan pengguna
                                    for(int z = 0; z < natt-1; z++)
                                    {
                                        System.out.println("Input attribute " + (z+1));
                                        float value = sc.nextFloat();
                                        inst.setValue(z, value);
                                    }
                                    if (discret) {
                                        disc.input(inst);
                                        inst = disc.output();
                                        double kelas = cls.classifyInstance(inst);
                                    } else {
                                        inst.setDataset(dataset);
                                    }
                                    double kelas = cls.classifyInstance(inst);
                                    System.out.println("Hasil: " + dataset.classAttribute().value((int) kelas));
                                    break;
                                default:
                                    break;
                            }
                        }
                        break;
                    default:
                        break;
                }
            }
        } else {
            
		long startTime = System.nanoTime();
		
		//Initialise
		BufferedReader br = null;
		Classifier cls = null;
		Instances train = null;
		Instances data = null;	
                Instances doto = null;
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
				cls = loadModel(name);
				//Input training file
				br = new BufferedReader(new FileReader("/home/chromeuser/Documents/student-train.arff"));
				data = new Instances(br); 
				br = new BufferedReader(new FileReader("/home/chromeuser/Documents/student-mat-test.arff"));
				doto = new Instances(br); 
				    
                                Remove remove = new Remove();
                                remove.setAttributeIndices("28");
                                remove.setInputFormat(data);
                                doto = Filter.useFilter(doto, remove);
                                train = Filter.useFilter(data, remove);
                                
                                train.setClassIndex(26);	
            					doto.setClassIndex(26);	
			} catch (Exception e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
		} else if(main==2){
			try {
				//Input training file
				br = new BufferedReader(new FileReader("/home/chromeuser/Documents/student-train.arff"));
				data = new Instances(br); 
				
				br = new BufferedReader(new FileReader("/home/chromeuser/Documents/student-mat-test.arff"));
				doto = new Instances(br);
				
                                
                                Remove remove = new Remove();
                                remove.setAttributeIndices("28");
                                remove.setInputFormat(data);
                                doto = Filter.useFilter(doto, remove);
                                train = Filter.useFilter(data, remove);
                                
                             
                     
				//Option variables
				String seed = "6196";
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
					train.setClassIndex(26);	
					doto.setClassIndex(26);	
					cls = new FFNN040();
					((FFNN040) cls).setOptions(cls_options);
					cls.buildClassifier(train);
				
			}catch (Exception e){
				e.printStackTrace();
			}
		}
		try {
			//Evaluate Model
			Evaluation eval = new Evaluation(train);
			System.out.println("");
			System.out.println("1. Full Training");
			System.out.println("2. 10-fold cross validation");
			System.out.println("Choose evaluation method :");
			int opt = in.nextInt();
			if(opt==1){
				eval.evaluateModel(cls, doto);
			} else if(opt==2){
				eval.crossValidateModel(cls, doto, 10, new Random(0));
			} else {
				System.out.println("invalid option");
			}
			in.nextLine();
			System.out.println();
			System.out.println(eval.toSummaryString("\nClassifier Loaded\n", true));
			
			long endTime = System.nanoTime();
			
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
}
