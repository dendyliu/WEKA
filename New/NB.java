/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package tucilweka;

/**
 *
 * @author HP
 */

import weka.classifiers.AbstractClassifier;
import weka.core.Attribute;
import weka.core.Capabilities;
import weka.core.Instance;
import weka.core.Instances;

public class NB extends AbstractClassifier {
 /*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
    
    int nAttributes;
    int nClasses;
    int nInstances;
    int clsIdx;
    double[] nClsVal;
    double[][][] matProb;

    @Override
    public Capabilities getCapabilities() {
      Capabilities result = super.getCapabilities();
      result.disableAll();

      // attributes
      result.enable(Capabilities.Capability.NOMINAL_ATTRIBUTES);
      result.enable(Capabilities.Capability.NUMERIC_ATTRIBUTES);
      result.enable( Capabilities.Capability.MISSING_VALUES );

      // class
      result.enable(Capabilities.Capability.NOMINAL_CLASS);
      result.enable(Capabilities.Capability.MISSING_CLASS_VALUES);

      // instances
      result.setMinimumNumberInstances(0);

      return result;
    }
    /**
     * @param args the command line arguments
     */
    

    @Override
    public void buildClassifier(Instances instances) throws Exception {
        // can classifier handle the data?
        getCapabilities().testWithFail(instances);

        // remove instances with missing class
        //instances = new Instances(instances);
        instances.deleteWithMissingClass();

        // Copy the instances
        //Instances m_Instances = new Instances(instances);

        // Reserve space for the distributions
        nAttributes = instances.numAttributes();
        nClasses = instances.numClasses();
        nInstances = instances.numInstances();
        Instance temp = instances.get(0);
        Attribute a = temp.attribute(1);
        int maxValues = 0;
        clsIdx = instances.classIndex();
        //Cari jumlah value terbesar dari semua atribut
        for(int i = 0; i<nAttributes; i++){
            int tempVal = instances.attribute(i).numValues();
            if(tempVal>maxValues){
                maxValues = tempVal;
            }
        }
        
        //insialisasi matrix
        matProb = new double[nAttributes][maxValues][nClasses];
        for(int i=0; i<nAttributes; i++) {
            if (i==clsIdx) {
                continue;
            }
            for(int j=0; j<maxValues; j++) {
                for(int k=0; k<nClasses; k++) {
                    matProb[i][j][k] = 0;
                }
            }
        }
        
        int lengthCls = instances.get(instances.classIndex()).numValues();
        nClsVal = new double[lengthCls];
        for(int i = 0; i<lengthCls; i++){
            nClsVal[i] = 0;
        }
        
        
        for(int i = 0; i<nInstances; i++){
            int classValue = (int)instances.get(i).classValue();
            nClsVal[classValue]++;
            for(int j = 0; j<nAttributes; j++){  
                if (j==clsIdx) {
                    continue;
                }
                int attr = (int)instances.get(i).value(j);
                matProb[j][attr][classValue] = matProb[j][attr][classValue] + 1.0; 
            }            
        }
        
        for(int i=0; i<nAttributes; i++) {
            if (i==clsIdx) {
                continue;
            }
            for(int j=0; j<maxValues; j++) {
                for(int k=0; k<nClasses; k++) {
                    matProb[i][j][k] = matProb[i][j][k]/nClsVal[k];
                }
            }
        }
        
        double total = 0;
        for(int i=0; i<nClasses; i++) {
            total += nClsVal[i];
        }
        
        for(int i=0; i<nClasses; i++) {
            nClsVal[i] /= total;
        }
    }
    
  @Override
  public double[] distributionForInstance(Instance instance) throws Exception {
    double[] prob = new double[nClasses];
    for (int i=0; i<nClasses; i++) {
        prob[i] = 1;
    }
    for (int i = 0; i<nAttributes; i++){
        if (i==clsIdx) {
            continue;
        }
        for(int j = 0; j<nClasses; j++){                
            int attr = (int) instance.value(i);
            prob[j] *= matProb[i][attr][j]; 
        }            
    }
    for (int i=0; i<nClasses; i++) {
        prob[i] *= nClsVal[i];
        System.out.println(prob[i]);
    }
    
    return prob;
  }
}
