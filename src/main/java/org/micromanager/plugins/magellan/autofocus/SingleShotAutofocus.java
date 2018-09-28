/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package main.java.org.micromanager.plugins.magellan.autofocus;

import java.awt.Point;
import java.io.File;
import java.nio.FloatBuffer;
import java.nio.file.FileSystem;
import java.nio.file.FileSystems;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.TreeMap;
import main.java.org.micromanager.plugins.magellan.acq.Acquisition;
import main.java.org.micromanager.plugins.magellan.acq.AcquisitionEvent;
import main.java.org.micromanager.plugins.magellan.acq.MagellanTaggedImage;
import main.java.org.micromanager.plugins.magellan.misc.GlobalSettings;
import main.java.org.micromanager.plugins.magellan.misc.Log;
import main.java.org.micromanager.plugins.magellan.misc.MD;
import org.tensorflow.SavedModelBundle;
import org.tensorflow.Session;
import org.tensorflow.Tensor;

/**
 *
 * @author Henry
 */
public class SingleShotAutofocus {

    private static SingleShotAutofocus singleton_ =null;

    SavedModelBundle smb_;
    private Session sess_;
    private String modelPath_;
    private String modelName_; //TODO get the short name
    private Acquisition cachedAcq_ = null;
    private TreeMap<Point, ArrayList<Double>> positonDefocusPredictions_ = new TreeMap<Point, ArrayList<Double>>();
            
    
    public SingleShotAutofocus() {
        singleton_ = this;
        //Try to load a model if one is remembered
        modelPath_ = GlobalSettings.getInstance().getStringInPrefs("Autofocus model path", null);
        try {
           if (modelPath_ != null) {        
              SavedModelBundle b = SavedModelBundle.load(modelPath_,"serve");
              sess_ = b.session();
              //String sep = "\\\\";
              //modelName_ = modelPath_.split(sep)[modelPath_.split(sep).length-1];
              modelName_ = modelPath_;
           }
        } catch (Exception e) {
           //problem loading autofocus model
           
        }
    }

    public double predictDefocus(MagellanTaggedImage img,  AcquisitionEvent event) {
      //init or reset previous predicitons cache
       if (cachedAcq_ == null){
          cachedAcq_ = event.acquisition_;
          
       } else if (event.acquisition_ != cachedAcq_){
          cachedAcq_ = event.acquisition_;
          positonDefocusPredictions_.clear();
       }        
       
       //predict from image quadrants
       Object[] quads = getImageQuadrants(img);
       ArrayList<Double> predictions = new ArrayList<Double>();
       for (Object q : quads) {
          double prediction = this.runModel((float[]) q);
          if (!Double.isNaN(prediction)) {
             predictions.add(prediction);
          }  
          System.out.println(prediction);
       }
       double[] preds = new double[predictions.size()];
       for (int i =0; i < preds.length; i++) {
           preds[i] = predictions.get(i);
       }

     
       //add all relevant nearby positions (i.e. previous column plus previous position
       int column = (int) event.xyPosition_.getGridCol();
       int row = (int) event.xyPosition_.getGridRow();
       ArrayList<Double> prevPredictions = new ArrayList<Double>();
      // if (positonDefocusPredictions_.containsKey(new Point(row, column - 1))) {
        //  prevPredictions.addAll(positonDefocusPredictions_.get(new Point(row, column - 1)));
      // }
      // if (positonDefocusPredictions_.containsKey(new Point(row-1, column - 1))) {
       //   prevPredictions.addAll(positonDefocusPredictions_.get(new Point(row -1, column - 1)));
     //  }       
       //if (positonDefocusPredictions_.containsKey(new Point(row+1 , column - 1))) {
      //    prevPredictions.addAll(positonDefocusPredictions_.get(new Point(row +1, column - 1)));
    //   }
       if (positonDefocusPredictions_.containsKey(new Point(row+1 , column))) {
          prevPredictions.addAll(positonDefocusPredictions_.get(new Point(row +1, column)));
       }
       if (positonDefocusPredictions_.containsKey(new Point(row-1, column))) {
          prevPredictions.addAll(positonDefocusPredictions_.get(new Point(row-1, column)));
       }
       
       //Combine all predicitons into one
       prevPredictions.addAll(predictions);
       double[] allPreds = new double[prevPredictions.size()];
       for (int i = 0; i < allPreds.length; i++){
          allPreds[i] = prevPredictions.get(i);
       } 
       
       //Take median
        Arrays.sort(allPreds);  
        double median;
        if (preds.length % 2 == 0) {
            median = ((double) preds[preds.length / 2] + (double) preds[preds.length / 2 - 1]) / 2;
        } else {
            median = (double) preds[preds.length / 2];
        }
        
       //add predictions to the cache
       positonDefocusPredictions_.put(new Point(row, column), predictions);
        
       return median;
    }

    private double runModel(float[] input) {
       
        long[] shape = new long[]{1,1024,1024};
        Tensor inputTensor = Tensor.create(shape, FloatBuffer.wrap(input));
         
       long start = System.currentTimeMillis();
       Tensor<Float> result = sess_.runner().feed("predict_input/input", inputTensor).fetch("predict_network/output").run().get(0).expect(Float.class);
//       System.out.println("Time to evaluate:" + (System.currentTimeMillis() - start) );
       
       float[] res=new float[1];
       result.copyTo(res); 
       double predictedDefocus = res[0];
       System.out.println("Total prediction: " + predictedDefocus);
       
      // Generally, there may be multiple output tensors, all of them must be closed to prevent resource leaks.
      //TODO: close all resources
              
       return predictedDefocus;
    }
    
    public static SingleShotAutofocus getInstance() {
       if (singleton_ == null) {
          singleton_ = new SingleShotAutofocus();
       }
       return singleton_;
    }

    public void loadModel(File f) {
       modelPath_ = f.getAbsolutePath();
       try{
         if (sess_ != null) {
            sess_.close();
         }
         if (smb_ != null) {
            smb_.close();
         }      
         smb_ = SavedModelBundle.load(modelPath_,"serve");
         sess_ = smb_.session();    
         GlobalSettings.getInstance().storeStringInPrefs("Autofocus model path", modelPath_);
         //String sep = "\\\\";
         //modelName_ = modelPath_.split(sep)[modelPath_.split(sep).length-1];
         modelName_ = modelPath_;
       } catch (Exception e) {
          Log.log(e);
       }
    }

    public String getModelName() {
       if (modelPath_ == null ) {
          return "No model loaded";
       } else {
          return modelName_;
       }
    }

    private Object[] getImageQuadrants(MagellanTaggedImage img) {
      int width = MD.getWidth(img.tags);
      short[] pixels = (short[]) img.pix;
      //divide in to 4 1024x1024 images
       
       //TODO: check orientation
       int quadDim = 1024;
       float[] topLeft = new float[quadDim*quadDim];
       float[] topRight = new float[quadDim*quadDim];
       float[] botLeft = new float[quadDim*quadDim];
       float[] botRight = new float[quadDim*quadDim];
       
       for (int x=0; x < 2*quadDim; x++ ) {
          for (int y=0; y < 2*quadDim; y++) {
             int destIndex = (x % quadDim) + (y % quadDim) * quadDim;
             int sourceIndex = x + y*width;
             if (x < quadDim && y < quadDim) {
                topLeft[destIndex] = pixels[sourceIndex]; 
             } else if (x < quadDim && y >= quadDim) {
                botLeft[destIndex] = pixels[sourceIndex]; 
             } else if (x >= quadDim && y < quadDim) {
                topRight[destIndex] = pixels[sourceIndex]; 
             } else if (x >= quadDim && y >= quadDim) {
                botRight[destIndex] = pixels[sourceIndex]; 
             }
          }
       }
       //visualize
//       FloatProcessor fp = new FloatProcessor(quadDim, quadDim, botLeft);
//       ImagePlus ip = new ImagePlus("test", fp);
//       ip.show();
       
       return new Object[]{topLeft, botLeft, topRight, botRight}; 
    }
}
