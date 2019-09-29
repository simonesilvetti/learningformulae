package com.eggloop.flow.simhya.simhya.matlab;

import com.eggloop.flow.expr.Context;
import com.eggloop.flow.expr.Variable;
import com.eggloop.flow.gaussianprocesses.gp.kernels.KernelRBF;
import com.eggloop.flow.gpoptimisation.gpoptim.GPOptimisation;
import com.eggloop.flow.gpoptimisation.gpoptim.GpoOptions;
import com.eggloop.flow.gpoptimisation.gpoptim.GpoResult;
import com.eggloop.flow.mitl.MiTL;
import com.eggloop.flow.mitl.MitlPropertiesList;
import com.eggloop.flow.model.Trajectory;
import com.eggloop.flow.numeric.optimization.ObjectiveFunction;
import com.eggloop.flow.parsers.MitlFactory;
import com.eggloop.flow.sampler.GridSampler;
import com.eggloop.flow.sampler.Parameter;
import com.eggloop.flow.simhya.simhya.matlab.genetic.*;
import com.eggloop.flow.utils.data.TrajectoryReconstruction;
import com.eggloop.flow.utils.files.Utils;

import java.util.Arrays;
import java.util.LinkedHashSet;
import java.util.Random;
import java.util.Set;
import java.util.stream.IntStream;

public class ComputeAverage {

    public static void main(String[] args) {
        Random ran = new Random(1);
        double[][] ds2SpatialValues = Utils.readMatrixFromFile(TestReading.class.getClassLoader().getResource("data/calin/ds2Trajectories").getPath());
        double[] ds2Labels = Utils.readVectorFromFile(TestReading.class.getClassLoader().getResource("data/calin/ds2Labels").getPath());
        double[] ds2Times = Utils.readVectorFromFile(TestReading.class.getClassLoader().getResource("data/calin/ds2Times").getPath());
        TrajectoryReconstruction data = new TrajectoryReconstruction(ds2Times, ds2SpatialValues, ds2Labels, 0.7, ran);
        data.split();
        double[][] normal_model = data.getPoistiveTrainSet();
        double[][] ineffective_model = data.getNegativeTrainSet();
        double[][] normal_model_test = data.getPoistiveValidationSet();
        double[][] ineffective_model_test = data.getNegativeValidationSet();


        FormulaPopulation pop = new FormulaPopulation(1);
        String[] variables = new String[]{"flow"};
        double[] lower = new double[]{0};
        double[] upper = new double[]{12};
        for (int i = 0; i < variables.length; i++) {
            pop.addVariable(variables[i], lower[i], upper[i]);
        }
        String[] pars = new String[]{"Tl_1", "Tu_1", "Tl_2", "Tu_2", "Tl_3", "Tu_3", "Theta_1", "Theta_2", "Theta_3"};
        double[] vv = new double[]{0.367, 52.2, 47, 96.6, 0.367, 52.2, 9.31, 9, 9.31};
        String FF = "P=?[ (G[Tl_1,Tu_1] {flow<=Theta_1} AND  G[Tl_2,Tu_2] {flow<=Theta_2}) OR  F[Tl_3,Tu_3] {flow >= Theta_3} ]";
        vv = new double[]{0.367, 52.2, 5, 7};
        pars = new String[]{"Tl_2", "Tu_2", "Theta_0", "Theta_1"};
        FF = "P=?[{flow >= Theta_1} U[Tl_2, Tu_2] {flow >= Theta_0}]";
        Formula formula = pop.loadFormula(FF, pars, vv);
        System.out.println(formula.toString());
        double[] res = average( ds2Times,  normal_model,  ineffective_model, formula,  pop,  new double[]{GeneticOptions.min_time_bound, GeneticOptions.max_time_bound});
        System.out.println(Arrays.toString(res));


        }

    public static double[] average(double[] ds2Times, double[][] normal_model, double[][] ineffective_model, Formula formula, FormulaPopulation pop,double[] timeBoundsFormula) {

        String[] variables = formula.getVariables();
        Set<String> set = new LinkedHashSet<>(Arrays.asList(variables ));
        final String[] variablesUnique = set.toArray(new String[set.size()]);
        //char[] a=formula.toSign().toCharArray();
        //System.out.println(formula.toString());



        String[] boundsFormula = formula.getTimeBounds();
        String[] pars = new String[boundsFormula.length + variables.length];
        String[] variablesFormula = formula.getVariables();
        String[] thresholdFormula = formula.getThresholds();
        double[] timeBoundsLb = Arrays.stream(boundsFormula).mapToDouble(x -> timeBoundsFormula[0]).toArray();
        double[] timeBoundsUb = Arrays.stream(boundsFormula).mapToDouble(x -> timeBoundsFormula[1]).toArray();
        double[] thrshldLb = Arrays.stream(variables).mapToDouble(pop::getLowerBound).toArray();
        double[] thrshldUb = Arrays.stream(variables).mapToDouble(pop::getUpperBound).toArray();
        double[] lb = new double[pars.length];
        double[] ub = new double[pars.length];
        System.arraycopy(timeBoundsLb, 0, lb, 0, timeBoundsLb.length);
        System.arraycopy(thrshldLb, 0, lb, boundsFormula.length, thrshldLb.length);
        System.arraycopy(timeBoundsUb, 0, ub, 0, timeBoundsUb.length);
        System.arraycopy(thrshldUb, 0, ub, boundsFormula.length, thrshldUb.length);


        ObjectiveFunction function = point -> {
            for (int i = 0; i < boundsFormula.length - 1; i += 2) {
                point[i + 1] = point[i] + point[i + 1] * (1 - point[i]);
            }
            final double[] p = point;
            point = IntStream.range(0, point.length).mapToDouble(i -> lb[i] + p[i] * (ub[i] - lb[i])).toArray();
//            double[] value1 = computeAverageRobustness(ds2Times, normal_model, variablesUnique, formula, concatenate(point, new double[]{(lb[point.length]+ub[point.length])/2}));
//            double[] value2 = computeAverageRobustness(ds2Times, ineffective_model, variablesUnique, formula, concatenate(point, new double[]{(lb[point.length]+ub[point.length])/2}));
            double[] value1 = computeAverageRobustness(ds2Times, normal_model, variablesUnique, formula, point);
            double[] value2 = computeAverageRobustness(ds2Times, ineffective_model, variablesUnique, formula, point);

            double abs = Math.abs((value1[0] - value2[0]) / (3 * (value1[1] + value2[1])));
            //System.out.println(abs);
            return abs;

        };

        GridSampler custom = new GridSampler() {
            @Override
            public double[][] sample(int n, double[] lbounds, double[] ubounds) {
                double[][] res = new double[n][lbounds.length];
                for (int i = 0; i < n; i++) {
                    for (int j = 0; j < boundsFormula.length; j += 2) {
                        res[i][j] = lbounds[j] + Math.random() * (ubounds[j] - lbounds[j]);
                        res[i][j + 1] = res[i][j] + (Math.random()) * (ubounds[j] - res[i][j]);
                    }
                    for (int j = boundsFormula.length; j < res[i].length; j++) {
                        res[i][j] = lbounds[j] + Math.random() * (ubounds[j] - lbounds[j]);
                    }
                }
                return res;
            }

            @Override
            public double[][] sample(int n, Parameter[] params) {
                return new double[0][];
            }
        };

        GPOptimisation gpo = new GPOptimisation();
        GpoOptions options = new GpoOptions();
        options.setInitialSampler(custom);
        //options.setGridSampler(new LatinHypercubeSampler(2, 10));
        options.setMaxIterations(200);
        options.setHyperparamOptimisation(true);
        options.setUseNoiseTermRatio(false);
        options.setGridSampler(custom);
        KernelRBF kernelGP = new KernelRBF();
        //kernelGP.setHyperarameters(new double[]{Math.log(normal_model.length + 1), 0.4});
        options.setKernelGP(kernelGP);
        //options.setGridSampleNumber(5);
        //options.setInitialObservations(2);
        gpo.setOptions(options);
        GpoResult optimise;
        double[] lbU = IntStream.range(0, lb.length ).mapToDouble(i -> 0).toArray();
        double[] ubU = IntStream.range(0, ub.length ).mapToDouble(i -> 1).toArray();
        optimise = gpo.optimise(function, lbU, ubU);
        // double[] v = optimise.getSolution();
        //System.out.println(Arrays.toString(v));
        //final double[] v = concatenate(optimise.getSolution(), new double[]{0.5});
        double[] v = optimise.getSolution();
        //vv= Arrays.stream(vv).map(v -> lb[0] + v * (ub[1] - lb[0])).toArray();
        double[] vv = IntStream.range(0, v.length).mapToDouble(i -> lb[i] + v[i] * (ub[i] - lb[i])).toArray();
        //System.out.println("àà"+Arrays.toString(vv));
        //System.out.println(Arrays.toString(optimise.getSolution()));

        double[] p1u1 = computeAverageRobustness(ds2Times, normal_model, variablesUnique, formula, vv);
        //System.out.println("OLD:" + Arrays.toString(p1u1));
        double[] p2u2 = computeAverageRobustness(ds2Times, ineffective_model, variablesUnique, formula, vv);
        //System.out.println("OLD:" + Arrays.toString(p2u2));
        double value;
        if (p1u1[0] > p2u2[0]) {
            value = ((p1u1[0] - 3 * p1u1[1]) + (p2u2[0] + 3 * p2u2[1])) / 2;
        } else {
            value = ((p2u2[0] - 3 * p2u2[1]) + (p1u1[0] + 3 * p1u1[1])) / 2;
        }
        // double value = (p1u1[0]+p2u2[0])/2;
        char[] a = formula.toSign().toCharArray();
        //System.out.println(a.length);
        //System.out.println(vv.length - timeBoundsLb.length);
        for (int i = timeBoundsLb.length; i < vv.length; i++) {
            if (a[i - timeBoundsLb.length] == '1') {
                vv[i] =Math.max(vv[i] + value, 0);
            } else {
                vv[i] = Math.max(vv[i] - value, 0);
            }
        }

        return vv;

    }
    public static double[] probMulti(String[] variablesUnique,double[] ds2Times, double[][][] normal_model, double[][][] ineffective_model, Formula formula, FormulaPopulation pop,double[] timeBoundsFormula) {

        String[] variables = formula.getVariables();
        Set<String> set = new LinkedHashSet<>(Arrays.asList(variables ));
        //final String[] variablesUnique = set.toArray(new String[set.size()]);
        //char[] a=formula.toSign().toCharArray();
        //System.out.println(formula.toString());



        String[] boundsFormula = formula.getTimeBounds();
        String[] pars = new String[boundsFormula.length + variables.length];
        String[] variablesFormula = formula.getVariables();
        String[] thresholdFormula = formula.getThresholds();
        double[] timeBoundsLb = Arrays.stream(boundsFormula).mapToDouble(x -> timeBoundsFormula[0]).toArray();
        double[] timeBoundsUb = Arrays.stream(boundsFormula).mapToDouble(x -> timeBoundsFormula[1]).toArray();
        double[] thrshldLb = Arrays.stream(variables).mapToDouble(pop::getLowerBound).toArray();
        double[] thrshldUb = Arrays.stream(variables).mapToDouble(pop::getUpperBound).toArray();
        double[] lb = new double[pars.length];
        double[] ub = new double[pars.length];
        System.arraycopy(timeBoundsLb, 0, lb, 0, timeBoundsLb.length);
        System.arraycopy(thrshldLb, 0, lb, boundsFormula.length, thrshldLb.length);
        System.arraycopy(timeBoundsUb, 0, ub, 0, timeBoundsUb.length);
        System.arraycopy(thrshldUb, 0, ub, boundsFormula.length, thrshldUb.length);


        ObjectiveFunction function = point -> {
            for (int i = 0; i < boundsFormula.length - 1; i += 2) {
                point[i + 1] = point[i] + point[i + 1] * (1 - point[i]);
            }
            final double[] p = point;
            point = IntStream.range(0, point.length).mapToDouble(i -> lb[i] + p[i] * (ub[i] - lb[i])).toArray();
//            double[] value1 = computeAverageRobustness(ds2Times, normal_model, variablesUnique, formula, concatenate(point, new double[]{(lb[point.length]+ub[point.length])/2}));
//            double[] value2 = computeAverageRobustness(ds2Times, ineffective_model, variablesUnique, formula, concatenate(point, new double[]{(lb[point.length]+ub[point.length])/2}));
            double[] value1 = computeProbMulti(ds2Times, normal_model, variablesUnique, formula, point);
            double[] value2 = computeProbMulti(ds2Times, ineffective_model, variablesUnique, formula, point);

            double abs = Math.max(Math.log(value1[0]+1/value2[0]+1),Math.log(value2[0]+1/value1[0]+1));
            //System.out.println(abs);
            if(Double.isNaN(abs)){
                return 0;
            }
            return abs;

        };

        GridSampler custom = new GridSampler() {
            @Override
            public double[][] sample(int n, double[] lbounds, double[] ubounds) {
                double[][] res = new double[n][lbounds.length];
                for (int i = 0; i < n; i++) {
                    for (int j = 0; j < boundsFormula.length; j += 2) {
                        res[i][j] = lbounds[j] + Math.random() * (ubounds[j] - lbounds[j]);
                        res[i][j + 1] = res[i][j] + (Math.random()) * (ubounds[j] - res[i][j]);
                    }
                    for (int j = boundsFormula.length; j < res[i].length; j++) {
                        res[i][j] = lbounds[j] + Math.random() * (ubounds[j] - lbounds[j]);
                    }
                }
                return res;
            }

            @Override
            public double[][] sample(int n, Parameter[] params) {
                return new double[0][];
            }
        };

        GPOptimisation gpo = new GPOptimisation();
        GpoOptions options = new GpoOptions();
        options.setInitialSampler(custom);
        //options.setGridSampler(new LatinHypercubeSampler(2, 10));
        options.setMaxIterations(50);
        options.setHyperparamOptimisation(true);
        options.setUseNoiseTermRatio(false);
        options.setGridSampler(custom);
        KernelRBF kernelGP = new KernelRBF();
        //kernelGP.setHyperarameters(new double[]{Math.log(normal_model.length + 1), 0.4});
        options.setKernelGP(kernelGP);
        //options.setGridSampleNumber(5);
        options.setInitialObservations(10);
        gpo.setOptions(options);
        GpoResult optimise;
        double[] lbU = IntStream.range(0, lb.length ).mapToDouble(i -> 0).toArray();
        double[] ubU = IntStream.range(0, ub.length ).mapToDouble(i -> 1).toArray();
        optimise = gpo.optimise(function, lbU, ubU);
        // double[] v = optimise.getSolution();
        //System.out.println(Arrays.toString(v));
        //final double[] v = concatenate(optimise.getSolution(), new double[]{0.5});
        double[] v = optimise.getSolution();
        //vv= Arrays.stream(vv).map(v -> lb[0] + v * (ub[1] - lb[0])).toArray();
        double[] vv = IntStream.range(0, v.length).mapToDouble(i -> lb[i] + v[i] * (ub[i] - lb[i])).toArray();
        //System.out.println("àà"+Arrays.toString(vv));
        //System.out.println(Arrays.toString(optimise.getSolution()));

//        double[] p1u1 = computeAverageRobustnessMulti(ds2Times, normal_model, variablesUnique, formula, vv);
//        //System.out.println("OLD:" + Arrays.toString(p1u1));
//        double[] p2u2 = computeAverageRobustnessMulti(ds2Times, ineffective_model, variablesUnique, formula, vv);
//        //System.out.println("OLD:" + Arrays.toString(p2u2));
//        double value;
//        if (p1u1[0] > p2u2[0]) {
//            value = ((p1u1[0] - 3 * p1u1[1]) + (p2u2[0] + 3 * p2u2[1])) / 2;
//        } else {
//            value = ((p2u2[0] - 3 * p2u2[1]) + (p1u1[0] + 3 * p1u1[1])) / 2;
//        }
//        // double value = (p1u1[0]+p2u2[0])/2;
//        char[] a = formula.toSign().toCharArray();
//        //System.out.println(a.length);
//        //System.out.println(vv.length - timeBoundsLb.length);
//        for (int i = timeBoundsLb.length; i < vv.length; i++) {
//            if (a[i - timeBoundsLb.length] == '1') {
//                vv[i] =Math.max(vv[i] + value, 0);
//            } else {
//                vv[i] = Math.max(vv[i] - value, 0);
//            }
//        }

        return vv;

    }

    public static double[] probMultiSuper(String[] variablesUnique,double[] ds2Times, double[][][] normal_model, double[][][] ineffective_model, Formula formula, FormulaPopulation pop,double[] timeBoundsFormula) {

        String[] variables = formula.getVariables();
        Set<String> set = new LinkedHashSet<>(Arrays.asList(variables ));
        //final String[] variablesUnique = set.toArray(new String[set.size()]);
        //char[] a=formula.toSign().toCharArray();
        //System.out.println(formula.toString());



        String[] boundsFormula = formula.getTimeBounds();
        String[] pars = new String[boundsFormula.length + variables.length];
        String[] variablesFormula = formula.getVariables();
        String[] thresholdFormula = formula.getThresholds();
        double[] timeBoundsLb = Arrays.stream(boundsFormula).mapToDouble(x -> timeBoundsFormula[0]).toArray();
        double[] timeBoundsUb = Arrays.stream(boundsFormula).mapToDouble(x -> timeBoundsFormula[1]).toArray();
        double[] thrshldLb = Arrays.stream(variables).mapToDouble(pop::getLowerBound).toArray();
        double[] thrshldUb = Arrays.stream(variables).mapToDouble(pop::getUpperBound).toArray();
        double[] lb = new double[pars.length];
        double[] ub = new double[pars.length];
        System.arraycopy(timeBoundsLb, 0, lb, 0, timeBoundsLb.length);
        System.arraycopy(thrshldLb, 0, lb, boundsFormula.length, thrshldLb.length);
        System.arraycopy(timeBoundsUb, 0, ub, 0, timeBoundsUb.length);
        System.arraycopy(thrshldUb, 0, ub, boundsFormula.length, thrshldUb.length);


        ObjectiveFunction function = point -> {
            for (int i = 0; i < boundsFormula.length - 1; i += 2) {
                point[i + 1] = point[i] + point[i + 1] * (1 - point[i]);
            }
            final double[] p = point;
            point = IntStream.range(0, point.length).mapToDouble(i -> lb[i] + p[i] * (ub[i] - lb[i])).toArray();
//            double[] value1 = computeAverageRobustness(ds2Times, normal_model, variablesUnique, formula, concatenate(point, new double[]{(lb[point.length]+ub[point.length])/2}));
//            double[] value2 = computeAverageRobustness(ds2Times, ineffective_model, variablesUnique, formula, concatenate(point, new double[]{(lb[point.length]+ub[point.length])/2}));
            double[] value1 = computeProbMulti(ds2Times, normal_model, variablesUnique, formula, point);
            double[] value2 = computeProbMulti(ds2Times, ineffective_model, variablesUnique, formula, point);

            double abs = Math.max(Math.log(value1[0]/value2[0]),Math.log(value2[0]/value1[0]));
            //System.out.println(abs);
            if(Double.isNaN(abs)){
                return 0;
            }
            return abs;

        };

        GridSampler custom = new GridSampler() {
            @Override
            public double[][] sample(int n, double[] lbounds, double[] ubounds) {
                double[][] res = new double[n][lbounds.length];
                for (int i = 0; i < n; i++) {
                    for (int j = 0; j < boundsFormula.length; j += 2) {
                        res[i][j] = lbounds[j] + Math.random() * (ubounds[j] - lbounds[j]);
                        res[i][j + 1] = res[i][j] + (Math.random()) * (ubounds[j] - res[i][j]);
                    }
                    for (int j = boundsFormula.length; j < res[i].length; j++) {
                        res[i][j] = lbounds[j] + Math.random() * (ubounds[j] - lbounds[j]);
                    }
                }
                return res;
            }

            @Override
            public double[][] sample(int n, Parameter[] params) {
                return new double[0][];
            }
        };

        GPOptimisation gpo = new GPOptimisation();
        GpoOptions options = new GpoOptions();
        options.setInitialSampler(custom);
        //options.setGridSampler(new LatinHypercubeSampler(2, 10));
        options.setMaxIterations(1600);
        options.setHyperparamOptimisation(true);
        options.setUseNoiseTermRatio(false);
        options.setGridSampler(custom);
        KernelRBF kernelGP = new KernelRBF();
        //kernelGP.setHyperarameters(new double[]{Math.log(normal_model.length + 1), 0.4});
        options.setKernelGP(kernelGP);
        //options.setGridSampleNumber(5);
        options.setInitialObservations(80);
        gpo.setOptions(options);
        GpoResult optimise;
        double[] lbU = IntStream.range(0, lb.length ).mapToDouble(i -> 0).toArray();
        double[] ubU = IntStream.range(0, ub.length ).mapToDouble(i -> 1).toArray();
        optimise = gpo.optimise(function, lbU, ubU);
        // double[] v = optimise.getSolution();
        //System.out.println(Arrays.toString(v));
        //final double[] v = concatenate(optimise.getSolution(), new double[]{0.5});
        double[] v = optimise.getSolution();
        //vv= Arrays.stream(vv).map(v -> lb[0] + v * (ub[1] - lb[0])).toArray();
        double[] vv = IntStream.range(0, v.length).mapToDouble(i -> lb[i] + v[i] * (ub[i] - lb[i])).toArray();
        //System.out.println("àà"+Arrays.toString(vv));
        //System.out.println(Arrays.toString(optimise.getSolution()));

        double[] p1u1 = computeAverageRobustnessMulti(ds2Times, normal_model, variablesUnique, formula, vv);
        //System.out.println("OLD:" + Arrays.toString(p1u1));
        double[] p2u2 = computeAverageRobustnessMulti(ds2Times, ineffective_model, variablesUnique, formula, vv);
        //System.out.println("OLD:" + Arrays.toString(p2u2));
        double value;
        if (p1u1[0] > p2u2[0]) {
            value = ((p1u1[0] - 3 * p1u1[1]) + (p2u2[0] + 3 * p2u2[1])) / 2;
        } else {
            value = ((p2u2[0] - 3 * p2u2[1]) + (p1u1[0] + 3 * p1u1[1])) / 2;
        }
        // double value = (p1u1[0]+p2u2[0])/2;
        char[] a = formula.toSign().toCharArray();
        //System.out.println(a.length);
        //System.out.println(vv.length - timeBoundsLb.length);
        for (int i = timeBoundsLb.length; i < vv.length; i++) {
            if (a[i - timeBoundsLb.length] == '1') {
                vv[i] =Math.max(vv[i] + value, 0);
            } else {
                vv[i] = Math.max(vv[i] - value, 0);
            }
        }

        return vv;

    }


    public static double[] averageMulti(String[] variablesUnique,double[] ds2Times, double[][][] normal_model, double[][][] ineffective_model, Formula formula, FormulaPopulation pop,double[] timeBoundsFormula) {

        String[] variables = formula.getVariables();
        Set<String> set = new LinkedHashSet<>(Arrays.asList(variables ));
        //final String[] variablesUnique = set.toArray(new String[set.size()]);
        //char[] a=formula.toSign().toCharArray();
        //System.out.println(formula.toString());



        String[] boundsFormula = formula.getTimeBounds();
        String[] pars = new String[boundsFormula.length + variables.length];
        String[] variablesFormula = formula.getVariables();
        String[] thresholdFormula = formula.getThresholds();
        double[] timeBoundsLb = Arrays.stream(boundsFormula).mapToDouble(x -> timeBoundsFormula[0]).toArray();
        double[] timeBoundsUb = Arrays.stream(boundsFormula).mapToDouble(x -> timeBoundsFormula[1]).toArray();
        double[] thrshldLb = Arrays.stream(variables).mapToDouble(pop::getLowerBound).toArray();
        double[] thrshldUb = Arrays.stream(variables).mapToDouble(pop::getUpperBound).toArray();
        double[] lb = new double[pars.length];
        double[] ub = new double[pars.length];
        System.arraycopy(timeBoundsLb, 0, lb, 0, timeBoundsLb.length);
        System.arraycopy(thrshldLb, 0, lb, boundsFormula.length, thrshldLb.length);
        System.arraycopy(timeBoundsUb, 0, ub, 0, timeBoundsUb.length);
        System.arraycopy(thrshldUb, 0, ub, boundsFormula.length, thrshldUb.length);


        ObjectiveFunction function = point -> {
            for (int i = 0; i < boundsFormula.length - 1; i += 2) {
                point[i + 1] = point[i] + point[i + 1] * (1 - point[i]);
            }
            final double[] p = point;
            point = IntStream.range(0, point.length).mapToDouble(i -> lb[i] + p[i] * (ub[i] - lb[i])).toArray();
//            double[] value1 = computeAverageRobustness(ds2Times, normal_model, variablesUnique, formula, concatenate(point, new double[]{(lb[point.length]+ub[point.length])/2}));
//            double[] value2 = computeAverageRobustness(ds2Times, ineffective_model, variablesUnique, formula, concatenate(point, new double[]{(lb[point.length]+ub[point.length])/2}));
            double[] value1 = computeAverageRobustnessMulti(ds2Times, normal_model, variablesUnique, formula, point);
            double[] value2 = computeAverageRobustnessMulti(ds2Times, ineffective_model, variablesUnique, formula, point);

            double abs = Math.abs((value1[0] - value2[0]) / (3 * (value1[1] + value2[1])));
            //System.out.println(abs);
            if(Double.isNaN(abs)){
                return 0;
            }
            return abs;

        };

        GridSampler custom = new GridSampler() {
            @Override
            public double[][] sample(int n, double[] lbounds, double[] ubounds) {
                double[][] res = new double[n][lbounds.length];
                for (int i = 0; i < n; i++) {
                    for (int j = 0; j < boundsFormula.length; j += 2) {
                        res[i][j] = lbounds[j] + Math.random() * (ubounds[j] - lbounds[j]);
                        res[i][j + 1] = res[i][j] + (Math.random()) * (ubounds[j] - res[i][j]);
                    }
                    for (int j = boundsFormula.length; j < res[i].length; j++) {
                        res[i][j] = lbounds[j] + Math.random() * (ubounds[j] - lbounds[j]);
                    }
                }
                return res;
            }

            @Override
            public double[][] sample(int n, Parameter[] params) {
                return new double[0][];
            }
        };

        GPOptimisation gpo = new GPOptimisation();
        GpoOptions options = new GpoOptions();
        options.setInitialSampler(custom);
        //options.setGridSampler(new LatinHypercubeSampler(2, 10));
        options.setMaxIterations(200);
        options.setHyperparamOptimisation(true);
        options.setUseNoiseTermRatio(false);
        options.setGridSampler(custom);
        KernelRBF kernelGP = new KernelRBF();
        //kernelGP.setHyperarameters(new double[]{Math.log(normal_model.length + 1), 0.4});
        options.setKernelGP(kernelGP);
        //options.setGridSampleNumber(5);
        //options.setInitialObservations(40);
        gpo.setOptions(options);
        GpoResult optimise;
        double[] lbU = IntStream.range(0, lb.length ).mapToDouble(i -> 0).toArray();
        double[] ubU = IntStream.range(0, ub.length ).mapToDouble(i -> 1).toArray();
        optimise = gpo.optimise(function, lbU, ubU);
        // double[] v = optimise.getSolution();
        //System.out.println(Arrays.toString(v));
        //final double[] v = concatenate(optimise.getSolution(), new double[]{0.5});
        double[] v = optimise.getSolution();
        //vv= Arrays.stream(vv).map(v -> lb[0] + v * (ub[1] - lb[0])).toArray();
        double[] vv = IntStream.range(0, v.length).mapToDouble(i -> lb[i] + v[i] * (ub[i] - lb[i])).toArray();
        //System.out.println("àà"+Arrays.toString(vv));
        //System.out.println(Arrays.toString(optimise.getSolution()));

        double[] p1u1 = computeAverageRobustnessMulti(ds2Times, normal_model, variablesUnique, formula, vv);
        //System.out.println("OLD:" + Arrays.toString(p1u1));
        double[] p2u2 = computeAverageRobustnessMulti(ds2Times, ineffective_model, variablesUnique, formula, vv);
        //System.out.println("OLD:" + Arrays.toString(p2u2));
        double value;
        if (p1u1[0] > p2u2[0]) {
            value = ((p1u1[0] - 3 * p1u1[1]) + (p2u2[0] + 3 * p2u2[1])) / 2;
        } else {
            value = ((p2u2[0] - 3 * p2u2[1]) + (p1u1[0] + 3 * p1u1[1])) / 2;
        }
        // double value = (p1u1[0]+p2u2[0])/2;
        char[] a = formula.toSign().toCharArray();
        //System.out.println(a.length);
        //System.out.println(vv.length - timeBoundsLb.length);
        for (int i = timeBoundsLb.length; i < vv.length; i++) {
            if (a[i - timeBoundsLb.length] == '1') {
                vv[i] =Math.max(vv[i] + value, 0);
            } else {
                vv[i] = Math.max(vv[i] - value, 0);
            }
        }

        return vv;

    }

    public static double[] averageMultiSuper(String[] variablesUnique,double[] ds2Times, double[][][] normal_model, double[][][] ineffective_model, Formula formula, FormulaPopulation pop,double[] timeBoundsFormula) {

        String[] variables = formula.getVariables();
        Set<String> set = new LinkedHashSet<>(Arrays.asList(variables ));
        //final String[] variablesUnique = set.toArray(new String[set.size()]);
        //char[] a=formula.toSign().toCharArray();
        //System.out.println(formula.toString());



        String[] boundsFormula = formula.getTimeBounds();
        String[] pars = new String[boundsFormula.length + variables.length];
        String[] variablesFormula = formula.getVariables();
        String[] thresholdFormula = formula.getThresholds();
        double[] timeBoundsLb = Arrays.stream(boundsFormula).mapToDouble(x -> timeBoundsFormula[0]).toArray();
        double[] timeBoundsUb = Arrays.stream(boundsFormula).mapToDouble(x -> timeBoundsFormula[1]).toArray();
        double[] thrshldLb = Arrays.stream(variables).mapToDouble(pop::getLowerBound).toArray();
        double[] thrshldUb = Arrays.stream(variables).mapToDouble(pop::getUpperBound).toArray();
        double[] lb = new double[pars.length];
        double[] ub = new double[pars.length];
        System.arraycopy(timeBoundsLb, 0, lb, 0, timeBoundsLb.length);
        System.arraycopy(thrshldLb, 0, lb, boundsFormula.length, thrshldLb.length);
        System.arraycopy(timeBoundsUb, 0, ub, 0, timeBoundsUb.length);
        System.arraycopy(thrshldUb, 0, ub, boundsFormula.length, thrshldUb.length);


        ObjectiveFunction function = point -> {
            for (int i = 0; i < boundsFormula.length - 1; i += 2) {
                point[i + 1] = point[i] + point[i + 1] * (1 - point[i]);
            }
            final double[] p = point;
            point = IntStream.range(0, point.length).mapToDouble(i -> lb[i] + p[i] * (ub[i] - lb[i])).toArray();
//            double[] value1 = computeAverageRobustness(ds2Times, normal_model, variablesUnique, formula, concatenate(point, new double[]{(lb[point.length]+ub[point.length])/2}));
//            double[] value2 = computeAverageRobustness(ds2Times, ineffective_model, variablesUnique, formula, concatenate(point, new double[]{(lb[point.length]+ub[point.length])/2}));
            double[] value1 = computeAverageRobustnessMulti(ds2Times, normal_model, variablesUnique, formula, point);
            double[] value2 = computeAverageRobustnessMulti(ds2Times, ineffective_model, variablesUnique, formula, point);

            double abs = Math.abs((value1[0] - value2[0]) / (3 * (value1[1] + value2[1])));
            //System.out.println(abs);
            if(Double.isNaN(abs)){
                return 0;
            }
            return abs;

        };

        GridSampler custom = new GridSampler() {
            @Override
            public double[][] sample(int n, double[] lbounds, double[] ubounds) {
                double[][] res = new double[n][lbounds.length];
                for (int i = 0; i < n; i++) {
                    for (int j = 0; j < boundsFormula.length; j += 2) {
                        res[i][j] = lbounds[j] + Math.random() * (ubounds[j] - lbounds[j]);
                        res[i][j + 1] = res[i][j] + (Math.random()) * (ubounds[j] - res[i][j]);
                    }
                    for (int j = boundsFormula.length; j < res[i].length; j++) {
                        res[i][j] = lbounds[j] + Math.random() * (ubounds[j] - lbounds[j]);
                    }
                }
                return res;
            }

            @Override
            public double[][] sample(int n, Parameter[] params) {
                return new double[0][];
            }
        };

        GPOptimisation gpo = new GPOptimisation();
        GpoOptions options = new GpoOptions();
        options.setInitialSampler(custom);
        //options.setGridSampler(new LatinHypercubeSampler(2, 10));
        options.setMaxIterations(1600);
        options.setHyperparamOptimisation(true);
        options.setUseNoiseTermRatio(false);
        options.setGridSampler(custom);
        KernelRBF kernelGP = new KernelRBF();
        //kernelGP.setHyperarameters(new double[]{Math.log(normal_model.length + 1), 0.4});
        options.setKernelGP(kernelGP);
        //options.setGridSampleNumber(100);
        options.setInitialObservations(60);
        gpo.setOptions(options);
        GpoResult optimise;
        double[] lbU = IntStream.range(0, lb.length ).mapToDouble(i -> 0).toArray();
        double[] ubU = IntStream.range(0, ub.length ).mapToDouble(i -> 1).toArray();
        optimise = gpo.optimise(function, lbU, ubU);
        // double[] v = optimise.getSolution();
        //System.out.println(Arrays.toString(v));
        //final double[] v = concatenate(optimise.getSolution(), new double[]{0.5});
        double[] v = optimise.getSolution();
        //vv= Arrays.stream(vv).map(v -> lb[0] + v * (ub[1] - lb[0])).toArray();
        double[] vv = IntStream.range(0, v.length).mapToDouble(i -> lb[i] + v[i] * (ub[i] - lb[i])).toArray();
        //System.out.println("àà"+Arrays.toString(vv));
        //System.out.println(Arrays.toString(optimise.getSolution()));

        double[] p1u1 = computeAverageRobustnessMulti(ds2Times, normal_model, variablesUnique, formula, vv);
        //System.out.println("OLD:" + Arrays.toString(p1u1));
        double[] p2u2 = computeAverageRobustnessMulti(ds2Times, ineffective_model, variablesUnique, formula, vv);
        //System.out.println("OLD:" + Arrays.toString(p2u2));
        double value;
        if (p1u1[0] > p2u2[0]) {
            value = ((p1u1[0] - 3 * p1u1[1]) + (p2u2[0] + 3 * p2u2[1])) / 2;
        } else {
            value = ((p2u2[0] - 3 * p2u2[1]) + (p1u1[0] + 3 * p1u1[1])) / 2;
        }
        // double value = (p1u1[0]+p2u2[0])/2;
        char[] a = formula.toSign().toCharArray();
        //System.out.println(a.length);
        //System.out.println(vv.length - timeBoundsLb.length);
        for (int i = timeBoundsLb.length; i < vv.length; i++) {
            if (a[i - timeBoundsLb.length] == '1') {
                vv[i] =Math.max(vv[i] + value, 0);
            } else {
                vv[i] = Math.max(vv[i] - value, 0);
            }
        }

        return vv;

    }

    public static double[] prob(double[] ds2Times, double[][] normal_model, double[][] ineffective_model, Formula formula, FormulaPopulation pop,double[] timeBoundsFormula) {

        String[] variables = formula.getVariables();
        Set<String> set = new LinkedHashSet<>(Arrays.asList(variables ));
        final String[] variablesUnique = set.toArray(new String[set.size()]);
        //char[] a=formula.toSign().toCharArray();
        //System.out.println(formula.toString());



        String[] boundsFormula = formula.getTimeBounds();
        String[] pars = new String[boundsFormula.length + variables.length];
        String[] variablesFormula = formula.getVariables();
        String[] thresholdFormula = formula.getThresholds();
        double[] timeBoundsLb = Arrays.stream(boundsFormula).mapToDouble(x -> timeBoundsFormula[0]).toArray();
        double[] timeBoundsUb = Arrays.stream(boundsFormula).mapToDouble(x -> timeBoundsFormula[1]).toArray();
        double[] thrshldLb = Arrays.stream(variables).mapToDouble(pop::getLowerBound).toArray();
        double[] thrshldUb = Arrays.stream(variables).mapToDouble(pop::getUpperBound).toArray();
        double[] lb = new double[pars.length];
        double[] ub = new double[pars.length];
        System.arraycopy(timeBoundsLb, 0, lb, 0, timeBoundsLb.length);
        System.arraycopy(thrshldLb, 0, lb, boundsFormula.length, thrshldLb.length);
        System.arraycopy(timeBoundsUb, 0, ub, 0, timeBoundsUb.length);
        System.arraycopy(thrshldUb, 0, ub, boundsFormula.length, thrshldUb.length);

        FitnessFunction regular = new RegularisedLogOddRatioFitness();
        ObjectiveFunction function = point -> {
            for (int i = 0; i < boundsFormula.length - 1; i += 2) {
                point[i + 1] = point[i] + point[i + 1] * (1 - point[i]);
            }
            final double[] p = point;
            point = IntStream.range(0, point.length).mapToDouble(i -> lb[i] + p[i] * (ub[i] - lb[i])).toArray();
//            double[] value1 = computeAverageRobustness(ds2Times, normal_model, variablesUnique, formula, concatenate(point, new double[]{(lb[point.length]+ub[point.length])/2}));
//            double[] value2 = computeAverageRobustness(ds2Times, ineffective_model, variablesUnique, formula, concatenate(point, new double[]{(lb[point.length]+ub[point.length])/2}));
            double[] value1 = computeProb(ds2Times, normal_model, variablesUnique, formula, point);
            double[] value2 = computeProb(ds2Times, ineffective_model, variablesUnique, formula, point);

            //double abs = Math.abs((value1[0] - value2[0]) / (3 * (value1[1] + value2[1])));
            //System.out.println(abs);
            return regular.compute(value1[0],value2[0],formula.getFormulaSize(),0,0,0);

        };

        GridSampler custom = new GridSampler() {
            @Override
            public double[][] sample(int n, double[] lbounds, double[] ubounds) {
                double[][] res = new double[n][lbounds.length];
                for (int i = 0; i < n; i++) {
                    for (int j = 0; j < boundsFormula.length; j += 2) {
                        res[i][j] = lbounds[j] + Math.random() * (ubounds[j] - lbounds[j]);
                        res[i][j + 1] = res[i][j] + (Math.random()) * (ubounds[j] - res[i][j]);
                    }
                    for (int j = boundsFormula.length; j < res[i].length; j++) {
                        res[i][j] = lbounds[j] + Math.random() * (ubounds[j] - lbounds[j]);
                    }
                }
                return res;
            }

            @Override
            public double[][] sample(int n, Parameter[] params) {
                return new double[0][];
            }
        };

        GPOptimisation gpo = new GPOptimisation();
        GpoOptions options = new GpoOptions();
        options.setInitialSampler(custom);
        //options.setGridSampler(new LatinHypercubeSampler(2, 10));
        //options.setMaxIterations(200);
        options.setHyperparamOptimisation(true);
        options.setUseNoiseTermRatio(false);
        options.setGridSampler(custom);
        KernelRBF kernelGP = new KernelRBF();
        //kernelGP.setHyperarameters(new double[]{Math.log(normal_model.length + 1), 0.4});
        options.setKernelGP(kernelGP);
        //options.setGridSampleNumber(5);
        //options.setInitialObservations(2);
        gpo.setOptions(options);
        GpoResult optimise;
        double[] lbU = IntStream.range(0, lb.length ).mapToDouble(i -> 0).toArray();
        double[] ubU = IntStream.range(0, ub.length ).mapToDouble(i -> 1).toArray();
        optimise = gpo.optimise(function, lbU, ubU);
        // double[] v = optimise.getSolution();
        //System.out.println(Arrays.toString(v));
        //final double[] v = concatenate(optimise.getSolution(), new double[]{0.5});
        double[] v = optimise.getSolution();
        //vv= Arrays.stream(vv).map(v -> lb[0] + v * (ub[1] - lb[0])).toArray();
        double[] vv = IntStream.range(0, v.length).mapToDouble(i -> lb[i] + v[i] * (ub[i] - lb[i])).toArray();
        //System.out.println("àà"+Arrays.toString(vv));
        //System.out.println(Arrays.toString(optimise.getSolution()));

//        double[] p1u1 = computeProb(ds2Times, normal_model, variablesUnique, formula, vv);
//        //System.out.println("OLD:" + Arrays.toString(p1u1));
//        double[] p2u2 = computeProb(ds2Times, ineffective_model, variablesUnique, formula, vv);
//        //System.out.println("OLD:" + Arrays.toString(p2u2));
//        double value;
//        if (p1u1[0] > p2u2[0]) {
//            value = ((p1u1[0] - 3 * p1u1[1]) + (p2u2[0] + 3 * p2u2[1])) / 2;
//        } else {
//            value = ((p2u2[0] - 3 * p2u2[1]) + (p1u1[0] + 3 * p1u1[1])) / 2;
//        }
//        // double value = (p1u1[0]+p2u2[0])/2;
//        char[] a = formula.toSign().toCharArray();
//        //System.out.println(a.length);
//        //System.out.println(vv.length - timeBoundsLb.length);
//        for (int i = timeBoundsLb.length; i < vv.length; i++) {
//            if (a[i - timeBoundsLb.length] == '1') {
//                vv[i] =Math.max(vv[i] + value, 0);
//            } else {
//                vv[i] = Math.max(vv[i] - value, 0);
//            }
//        }

        return vv;

    }


    public static double[] computeAverageRobustness(double[] times, double[][] simulate, String[] variables, Formula formula, double[] vv) {
        double[] b = new double[simulate.length];
        Context ns = new Context();
        for (String s : variables) {
            new Variable(s, ns);
        }
        String[] parameters = formula.getParameters();
        StringBuilder builder = new StringBuilder();
       // builder.append("const double infinity = ")
        for (int j = 0; j < parameters.length; j++) {
            builder.append("const double ").append(parameters[j]).append("=").append(vv[j]).append(";\n");
        }
        builder.append(formula.toString() + "\n");

        //builder.append("((G[Tl_1, Tu_1] (flow <= Theta_1) & G[Tl_2, Tu_2] (flow <= Theta_2) )  | F[Tl_3, Tu_3] (flow >= Theta_3))\n");
        // builder.append("((G[Tl_1, Tu_1] (flow <= Theta_1) & G[Tl_2, Tu_2] (flow <= Theta_2))  | F[Tl_3, Tu_3] (flow >= Theta_3))\n");
        MitlFactory factory = new MitlFactory(ns);
        String text = builder.toString();
        //System.out.println(text);
        //System.out.println(text);
        MitlPropertiesList l = factory.constructProperties(text);
        MiTL prop = l.getProperties().get(0);
        //System.out.println(text);
        for (int i = 0; i < simulate.length; i++) {
            Trajectory x = new Trajectory(times, ns, new double[][]{simulate[i]});
            b[i] = prop.evaluateValue(x, 0);
        }
        //F[Tl_213, Tu_213] (G[Tl_212, Tu_212] (((F[Tl_215, Tu_215] (flow <= Theta_214)) U[Tl_217, Tu_217] (flow <= Theta_216))) )
        double mean = Arrays.stream(b).sum() / b.length;
        double sigma = Math.sqrt(Arrays.stream(b).map(x -> (x - mean) * (x - mean)).sum() / b.length);
//        double min = Arrays.stream(b).min().getAsDouble();
//        double max = Arrays.stream(b).max().getAsDouble();


        return new double[]{mean, sigma};

    }

    public static double[] computeAverageRobustnessMulti(double[] times, double[][][] simulate, String[] variables, Formula formula, double[] parametersValues) {
        double[] b = new double[simulate.length];
        Context ns = new Context();
        for (String s : variables) {
            new Variable(s, ns);
        }
        String[] parameters = formula.getParameters();
        StringBuilder builder = new StringBuilder();
        // builder.append("const double infinity = ")
        for (int j = 0; j < parameters.length; j++) {
            builder.append("const double ").append(parameters[j]).append("=").append(parametersValues[j]).append(";\n");
        }
        builder.append(formula.toString() + "\n");

        //builder.append("((G[Tl_1, Tu_1] (flow <= Theta_1) & G[Tl_2, Tu_2] (flow <= Theta_2) )  | F[Tl_3, Tu_3] (flow >= Theta_3))\n");
        // builder.append("((G[Tl_1, Tu_1] (flow <= Theta_1) & G[Tl_2, Tu_2] (flow <= Theta_2))  | F[Tl_3, Tu_3] (flow >= Theta_3))\n");
        MitlFactory factory = new MitlFactory(ns);
        String text = builder.toString();
        //System.out.println(text);
        //System.out.println(text);
        MitlPropertiesList l = factory.constructProperties(text);
        MiTL prop = l.getProperties().get(0);
        //System.out.println(text);
        for (int i = 0; i < simulate.length; i++) {
            Trajectory x = new Trajectory(times, ns, simulate[i]);
            b[i] = prop.evaluateValue(x, 0);
        }
        //F[Tl_213, Tu_213] (G[Tl_212, Tu_212] (((F[Tl_215, Tu_215] (flow <= Theta_214)) U[Tl_217, Tu_217] (flow <= Theta_216))) )
        double mean = Arrays.stream(b).sum() / b.length;
        double sigma = Math.sqrt(Arrays.stream(b).map(x -> (x - mean) * (x - mean)).sum() / b.length);
//        double min = Arrays.stream(b).min().getAsDouble();
//        double max = Arrays.stream(b).max().getAsDouble();


        return new double[]{mean, sigma};

    }

    public static double[] computeProbMulti(double[] times, double[][][] simulate, String[] variables, Formula formula, double[] vv) {
        int[] b = new int[simulate.length];
        Context ns = new Context();
        for (String s : variables) {
            new Variable(s, ns);
        }
        String[] parameters = formula.getParameters();
        StringBuilder builder = new StringBuilder();
        // builder.append("const double infinity = ")
        for (int j = 0; j < parameters.length; j++) {
            builder.append("const double ").append(parameters[j]).append("=").append(vv[j]).append(";\n");
        }
        builder.append(formula.toString() + "\n");

        //builder.append("((G[Tl_1, Tu_1] (flow <= Theta_1) & G[Tl_2, Tu_2] (flow <= Theta_2) )  | F[Tl_3, Tu_3] (flow >= Theta_3))\n");
        // builder.append("((G[Tl_1, Tu_1] (flow <= Theta_1) & G[Tl_2, Tu_2] (flow <= Theta_2))  | F[Tl_3, Tu_3] (flow >= Theta_3))\n");
        MitlFactory factory = new MitlFactory(ns);
        String text = builder.toString();
        //System.out.println(text);
        //System.out.println(text);
        MitlPropertiesList l = factory.constructProperties(text);
        MiTL prop = l.getProperties().get(0);
        //System.out.println(text);
        for (int i = 0; i < simulate.length; i++) {
            Trajectory x = new Trajectory(times, ns, simulate[i]);
            b[i] = prop.evaluate(x, 0)?1:0;
        }
        //F[Tl_213, Tu_213] (G[Tl_212, Tu_212] (((F[Tl_215, Tu_215] (flow <= Theta_214)) U[Tl_217, Tu_217] (flow <= Theta_216))) )
        double mean = Arrays.stream(b).sum() / (double)b.length;
        double sigma = 1.0-mean;
//        double min = Arrays.stream(b).min().getAsDouble();
//        double max = Arrays.stream(b).max().getAsDouble();


        return new double[]{mean, sigma};

    }


    public static double[] computeProb(double[] times, double[][] simulate, String[] variables, Formula formula, double[] vv) {
        int[] b = new int[simulate.length];
        Context ns = new Context();
        for (String s : variables) {
            new Variable(s, ns);
        }
        String[] parameters = formula.getParameters();
        StringBuilder builder = new StringBuilder();
        // builder.append("const double infinity = ")
        for (int j = 0; j < parameters.length; j++) {
            builder.append("const double ").append(parameters[j]).append("=").append(vv[j]).append(";\n");
        }
        builder.append(formula.toString() + "\n");

        //builder.append("((G[Tl_1, Tu_1] (flow <= Theta_1) & G[Tl_2, Tu_2] (flow <= Theta_2) )  | F[Tl_3, Tu_3] (flow >= Theta_3))\n");
        // builder.append("((G[Tl_1, Tu_1] (flow <= Theta_1) & G[Tl_2, Tu_2] (flow <= Theta_2))  | F[Tl_3, Tu_3] (flow >= Theta_3))\n");
        MitlFactory factory = new MitlFactory(ns);
        String text = builder.toString();
        //System.out.println(text);
        //System.out.println(text);
        MitlPropertiesList l = factory.constructProperties(text);
        MiTL prop = l.getProperties().get(0);
        //System.out.println(text);
        for (int i = 0; i < simulate.length; i++) {
            Trajectory x = new Trajectory(times, ns, new double[][]{simulate[i]});
            b[i] = prop.evaluate(x, 0)?1:0;
        }
        //F[Tl_213, Tu_213] (G[Tl_212, Tu_212] (((F[Tl_215, Tu_215] (flow <= Theta_214)) U[Tl_217, Tu_217] (flow <= Theta_216))) )
        double buoni = Arrays.stream(b).count() / b.length;
        double cattivi = b.length-buoni;
//        double min = Arrays.stream(b).min().getAsDouble();
//        double max = Arrays.stream(b).max().getAsDouble();


        return new double[]{buoni, cattivi};

    }

}


