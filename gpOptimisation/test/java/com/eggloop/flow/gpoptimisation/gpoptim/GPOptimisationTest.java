package com.eggloop.flow.gpoptimisation.gpoptim;

import com.eggloop.flow.numeric.optimization.ObjectiveFunction;
import com.eggloop.flow.numeric.optimization.methods.ConjugateGradientApache;
import com.eggloop.flow.numeric.optimization.methods.PowellMethodApache;
import org.junit.Test;

import java.util.Arrays;

public class GPOptimisationTest {

    @Test
    public void optimise() {
        ObjectiveFunction fun = point -> Math.cos(point[0]) * point[1];
        GPOptimisation optimisation = new GPOptimisation();
        GpoOptions options = new GpoOptions();
        options.setHyperparamOptimisation(true);
        options.setUseNoiseTermRatio(true);
        options.setNoiseTerm(0);
        options.setLocalOptimiser(new PowellMethodApache());
//        options.setInitialObservations(60);
        optimisation.setOptions(options);
        GpoResult optimise = optimisation.optimise(fun, new double[]{0, 1}, new double[]{10, 89});
        System.out.println(Arrays.toString(optimise.getSolution()));
        System.out.println(optimise.getTerminationCause());
        System.out.println(optimise.getEvaluations());
        System.out.println(optimise.getHyperparamOptimTimeElapsed());
    }

    @Test
    public void optimise1() {
    }
}