package com.eggloop.flow.gaussianprocesses.prove;

import org.apache.commons.math3.analysis.MultivariateFunction;
import org.apache.commons.math3.optim.*;
import org.apache.commons.math3.optim.nonlinear.scalar.MultivariateFunctionMappingAdapter;
import org.apache.commons.math3.optim.nonlinear.scalar.MultivariateOptimizer;
import org.apache.commons.math3.optim.nonlinear.scalar.ObjectiveFunction;
import org.apache.commons.math3.optim.nonlinear.scalar.noderiv.PowellOptimizer;

import java.util.Arrays;

/**
 * Created by simone on 23/12/16.
 */
public class Prova {

    public static void main(String[] args) {
        MultivariateFunction f = doubles -> {
            System.out.println(Arrays.toString(doubles));
//            double[] v  = new double[]{1,2,3};
//            return -IntStream.range(0,v.length).mapToDouble(s -> doubles[s]-v[s]).map(s -> s*s).sum();
            return Arrays.stream(doubles).sum();
        };
        MultivariateFunctionMappingAdapter ff = new MultivariateFunctionMappingAdapter(f,new double[]{1,2,3},new double[]{4,7,8});

        PointValuePair minimum = findMinimum(ff);
        System.out.printf(Arrays.toString(ff.unboundedToBounded(minimum.getPoint())));

    }

    public static PointValuePair findMinimum (MultivariateFunction f){
        MultivariateOptimizer optim = new PowellOptimizer(1e-8, 1e-8);
        final OptimizationData[] optimData = new OptimizationData[3];
        optimData[0] = new ObjectiveFunction(f);
        optimData[1] = new MaxEval(500);
        optimData[2] = new InitialGuess(new double[]{2,3,5});
//        optimData[3] = new SimpleBounds(new double[]{0,0,0}, new double[]{5,7,8});

        //optimData[2] = new InitialGuess(init);
        PointValuePair pair = optim.optimize(optimData);


        return pair;


    }

}
