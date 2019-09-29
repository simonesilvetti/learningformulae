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
import com.eggloop.flow.simhya.simhya.matlab.genetic.*;
import com.eggloop.flow.utils.data.Bootstrap;
import com.eggloop.flow.utils.data.TrajectoryReconstruction;
import com.eggloop.flow.utils.files.Utils;

import java.util.Arrays;
import java.util.Random;
import java.util.stream.IntStream;

public class TestReading {
    public static void main(String[] args) {
        Random ran = new Random(1);
        double[][] ds2SpatialValues = Utils.readMatrixFromFile(TestReading.class.getClassLoader().getResource("data/calin/ds2Trajectories").getPath());
        double[] ds2Labels = Utils.readVectorFromFile(TestReading.class.getClassLoader().getResource("data/calin/ds2Labels").getPath());
        double[] ds2Times = Utils.readVectorFromFile(TestReading.class.getClassLoader().getResource("data/calin/ds2Times").getPath());
        TrajectoryReconstruction data = new TrajectoryReconstruction(ds2Times, ds2SpatialValues, ds2Labels, 0.1,ran);
        data.split();
        double[][] normal_model = data.getPoistiveTrainSet();
        double[][] ineffective_model = data.getNegativeTrainSet();
        double[][] normal_model_test = data.getPoistiveValidationSet();
        double[][] ineffective_model_test = data.getNegativeValidationSet();

        GeneticOptions.setMax_time_bound(100);
        GeneticOptions.setMin_time_bound(0);

        FormulaPopulation pop = new FormulaPopulation(1);
        String[] variables = new String[]{"flow"};
        double[] lower = new double[]{0};
        double[] upper = new double[]{12};
        for (int i = 0; i < variables.length; i++) {
            pop.addVariable(variables[i], lower[i], upper[i]);
        }
        //String[] pars = new String[]{"Tl_1", "Tu_1", "Tl_2", "Tu_2", "Tl_3", "Tu_3", "Theta_1", "Theta_2", "Theta_3"};
        //double[] vv = new double[]{10, 15, 3, 8, 0.367, 52.2, 3, 12, 4};
       // String FF = "P=?[ (G[Tl_1,Tu_1] {flow<=Theta_1} AND  G[Tl_2,Tu_2] {flow<=Theta_2}) OR  F[Tl_3,Tu_3] {flow >= Theta_3} ]";
        String[] pars = new String[]{"Tl_3", "Tu_3","Theta_3"};
        double[] vv = new double[]{ 0.367, 52.2, 4};
        String FF = "P=?[F[Tl_3,Tu_3] {flow >= Theta_3}]";
        Formula formula = pop.loadFormula(FF, pars, vv);


//        Random ran = new Random(1);
//        double[][] normal_model_boot = Bootstrap.bootstrap(normal_model.length,normal_model, ran);
//        double[][] ineffective_model_boot = Bootstrap.bootstrap(ineffective_model.length,ineffective_model, ran);
//

        double[] bestPar = continuousOptimization(pop, formula, normal_model, ineffective_model, ds2Times);
        bestPar = reconvertTimeBounds(formula.getParameters(), bestPar, new double[]{GeneticOptions.min_time_bound, GeneticOptions.max_time_bound});
        double[] p1u1 = smc(pop, formula,bestPar, ds2Times, normal_model_test);
        double[] p2u2 = smc(pop, formula,bestPar, ds2Times, ineffective_model_test);
        System.out.println(Arrays.toString(bestPar));
        System.out.println("Optimized: " + Arrays.toString(p1u1));
        System.out.println("Optimized: " + Arrays.toString(p2u2));

        FormulaPopulation pop2 = new FormulaPopulation(1);
         variables = new String[]{"flow"};
         lower = new double[]{0};
        upper = new double[]{12};
        for (int i = 0; i < variables.length; i++) {
            pop2.addVariable(variables[i], lower[i], upper[i]);
        }
//         pars = new String[]{"Tl_1", "Tu_1", "Tl_2", "Tu_2", "Tl_3", "Tu_3", "Theta_1", "Theta_2", "Theta_3"};
//         vv = new double[]{10, 15, 3, 8, 0.367, 52.2, 3, 12, 4};
//        FF = "P=?[ (G[Tl_1,Tu_1] {flow<=Theta_1} AND  G[Tl_2,Tu_2] {flow<=Theta_2}) OR  F[Tl_3,Tu_3] {flow >= Theta_3} ]";
//        formula = pop2.loadFormula(FF, pars, vv);
        //(G[Tl_1, Tu_1] ((flow <= Theta_1 & G[Tl_2, Tu_2] (flow <= Theta_2))) | F[Tl_3, Tu_3] (flow >= Theta_3))
        pars = new String[]{"Tl_3", "Tu_3","Theta_3"};
        vv = new double[]{ 0.367, 52.2, 4};
        FF = "P=?[F[Tl_3,Tu_3] {flow >= Theta_3}]";
        formula = pop2.loadFormula(FF, pars, vv);


        double[] bestPar2 = continuousOptimizationRobustness(pop2,variables, formula, normal_model, ineffective_model, ds2Times);
        bestPar2 = reconvertTimeBounds(formula.getParameters(), bestPar2, new double[]{GeneticOptions.min_time_bound, GeneticOptions.max_time_bound});
        p1u1 = smc(pop2, formula,bestPar2, ds2Times, normal_model_test);
        p2u2 = smc(pop2, formula,bestPar2, ds2Times, ineffective_model_test);
        System.out.println(Arrays.toString(bestPar2));
        System.out.println("Optimized_wrobustness: " + Arrays.toString(p1u1));
        System.out.println("Optimized_wrobustness: " + Arrays.toString(p2u2));


//        double[] p1u1 = smc(pop, formula,ds2Times, normal_model);
//        double[] p2u2 = smc(pop, formula,ds2Times, ineffective_model);
//        System.out.println(Arrays.toString(p1u1));
//        System.out.println(Arrays.toString(p2u2));
//        FitnessFunction fitness = new RegularisedLogOddRatioFitness();
//        //double value = fitness.compute(p1u1[0], p2u2[0], formula.getFormulaSize(), p1u1[1], p2u2[1], normal_model.length);
//        double value = Math.max(Math.log(p1u1[0]/p2u2[0]),Math.log(p2u2[0]/p1u1[0]));
//        System.out.println(value);
//
//        Random ran = new Random();
//        int N=1;
//        double[][] paramMatrix = new double[N][];
//        for (int i = 0; i < N; i++) {
//            //ran.setSeed(i);
//            vv = new double[]{10, 15, 3, 8, 0.367, 52.2, 9.31, 9, 9.31};
//            formula = pop.loadFormula(FF, pars, vv);
//            double[][] normal_model_boot = Bootstrap.bootstrap(2*normal_model.length,normal_model, ran);
//            double[][] ineffective_model_boot = Bootstrap.bootstrap(2*ineffective_model.length,ineffective_model, ran);
//            double[] bestPar = continuousOptimization(pop, formula, normal_model_boot, ineffective_model_boot, ds2Times);
//            bestPar = reconvertTimeBounds(formula.getParameters(), bestPar, new double[]{GeneticOptions.min_time_bound, GeneticOptions.max_time_bound});
//            paramMatrix[i] = bestPar;
//        }
//
//        double[] p1u1 =smcBoot(pop, formula, paramMatrix,ds2Times, normal_model_test);
//        double[] p2u2 = smcBoot(pop, formula,  paramMatrix,ds2Times, ineffective_model_test);
//        System.out.println("BootStrap: " + Arrays.toString(p1u1));
//        System.out.println("Bootstrap: " + Arrays.toString(p2u2));
//
//        vv = new double[]{10, 15, 3, 8, 0.367, 52.2, 9.31, 9, 9.31};
//        formula = pop.loadFormula(FF, pars, vv);
//
//        double[] bestPar = continuousOptimization(pop, formula, normal_model, ineffective_model, ds2Times);
//        bestPar = reconvertTimeBounds(formula.getParameters(), bestPar, new double[]{GeneticOptions.min_time_bound, GeneticOptions.max_time_bound});
//
//        //formula = pop.loadFormula(FF, pars, bestPar);
//        //System.out.println(Arrays.toString(bestPar));
//        p1u1 = smc(pop, formula,bestPar, ds2Times, normal_model_test);
//        p2u2 = smc(pop, formula,bestPar, ds2Times, ineffective_model_test);
//        System.out.println("Optimized: " + Arrays.toString(p1u1));
//        System.out.println("Optimized: " + Arrays.toString(p2u2));

        pop = new FormulaPopulation(1);
        variables = new String[]{"flow"};
        lower = new double[]{0};
        upper = new double[]{12};
        for (int i = 0; i < variables.length; i++) {
            pop.addVariable(variables[i], lower[i], upper[i]);
        }
//        pars = new String[]{"Tl_1", "Tu_1", "Tl_2", "Tu_2", "Tl_3", "Tu_3", "Theta_1", "Theta_2", "Theta_3"};
//        FF = "P=?[ (G[Tl_1,Tu_1] {flow<=Theta_1} AND  G[Tl_2,Tu_2] {flow<=Theta_2}) OR  F[Tl_3,Tu_3] {flow >= Theta_3} ]";
//        vv = new double[]{0.367, 52.2, 47, 96.6, 0.367, 52.2, 9.31, 9, 9.31};
//        formula = pop.loadFormula(FF, pars, vv);


        pars = new String[]{"Tl_3", "Tu_3","Theta_3"};
        vv = new double[]{ 0.367, 52.2, 9.31};
        FF = "P=?[F[Tl_3,Tu_3] {flow >= Theta_3}]";
        formula = pop.loadFormula(FF, pars, vv);

        p1u1 = smc(pop, formula,vv, ds2Times, normal_model_test);
        p2u2 = smc(pop, formula,vv, ds2Times, ineffective_model_test);
        System.out.println(Arrays.toString(vv));
        System.out.println("Calin: " + Arrays.toString(p1u1));
        System.out.println("Calin: " + Arrays.toString(p2u2));

        pars = new String[]{"Tl_3", "Tu_3","Theta_3"};
        vv = new double[]{ 0.367, 52.2, 9.31};
        FF = "P=?[F[Tl_3,Tu_3] {flow >= Theta_3}]";
        formula = pop.loadFormula(FF, pars, vv);

        p1u1 = smcAverage(pop, formula,variables, ds2Times, normal_model_test,vv);
        p2u2 = smcAverage(pop, formula,variables, ds2Times, ineffective_model_test,vv);
        System.out.println(Arrays.toString(vv));
        System.out.println("rob: " + Arrays.toString(p1u1));
        System.out.println("rob: " + Arrays.toString(p2u2));
        FitnessFunction fit = new NewFitness();
        System.out.println(fit.compute(p1u1[0],p2u2[0],0,p1u1[1],p2u2[1],0));

        pars = new String[]{"Tl_3", "Tu_3","Theta_3"};
        vv = new double[]{0.0, 85.44871670422907, 8.3075160834352};

        FF = "P=?[F[Tl_3,Tu_3] {flow >= Theta_3}]";
        formula = pop.loadFormula(FF, pars, vv);

        p1u1 = smcAverage(pop, formula,variables, ds2Times, normal_model_test,vv);
        p2u2 = smcAverage(pop, formula,variables, ds2Times, ineffective_model_test,vv);
        System.out.println(Arrays.toString(vv));
        System.out.println("rob_brutto: " + Arrays.toString(p1u1));
        System.out.println("rob_brutto: " + Arrays.toString(p2u2));
         fit = new NewFitness();
        System.out.println(fit.compute(p1u1[0],p2u2[0],0,p1u1[1],p2u2[1],0));

    }


    static double[] smc(FormulaPopulation popgen, Formula formula, double[] parameters, double[] times, double[][] trajectories) {
        int[] data = new int[trajectories.length];
        for (int i = 0; i < trajectories.length; i++) {
            popgen.setParameters(formula.getParameters(), parameters);
            data[i] = popgen.modelCheck(combineTimeAndSpace(times, new double[][]{trajectories[i]}), formula);
        }
        double p = (double) (Arrays.stream(data).filter(x -> x == 1).count() + 1) / (trajectories.length + 2);
        double u = (double) (Arrays.stream(data).filter(x -> x == -1).count()) / (trajectories.length);
//        p = (sum(data==1)+1)/(samples+2);
//        u = (sum(data==-1))/(samples)
        return new double[]{p, u};

    }

    static double[] smcBoot(FormulaPopulation popgen, Formula formula, double[][] parameters, double[] times, double[][] trajectories) {
        int[] data = new int[trajectories.length];
        String FF = "P=?[ (G[Tl_1,Tu_1] {flow<=Theta_1} AND  G[Tl_2,Tu_2] {flow<=Theta_2}) OR  F[Tl_3,Tu_3] {flow >= Theta_3} ]";

        for (int i = 0; i < trajectories.length; i++) {
            double[] appo = new double[trajectories.length];
           // formula = popgen.loadFormula(FF,formula.getParameters(), parameters[i]);
            for (int j = 0; j < parameters.length; j++) {
                popgen.setParameters(formula.getParameters(), parameters[j]);
                appo[j] = popgen.modelCheck(combineTimeAndSpace(times, new double[][]{trajectories[i]}), formula);
            }
            double value = (double) (Arrays.stream(appo).filter(x -> x == 1).count()) / (parameters.length);
            data[i] = value > 0.5 ? 1 : 0;
        }
        double p = (double) (Arrays.stream(data).filter(x -> x == 1).count() + 1) / (trajectories.length + 2);
        double u = (double) (Arrays.stream(data).filter(x -> x == -1).count()) / (trajectories.length);
//        p = (sum(data==1)+1)/(samples+2);
//        u = (sum(data==-1))/(samples)
        return new double[]{p, u};

    }

    private static double[][] combineTimeAndSpace(double[] times, double[][] spaceTrajectories) {
        double[][] res = new double[spaceTrajectories.length + 1][];
        res[0] = times;
        System.arraycopy(spaceTrajectories, 0, res, 1, spaceTrajectories.length);
        return res;

    }


    static double[] continuousOptimization(FormulaPopulation popgen, Formula formula, double[][] positiveDataset, double[][] negativeDataset, double[] times) {
        double[] timeBounds = new double[]{GeneticOptions.min_time_bound, GeneticOptions.max_time_bound};
        String[] bounds = formula.getTimeBounds();
        String[] variables = formula.getVariables();
        String[] trshld = formula.getThresholds();
        String[] pars = new String[bounds.length + variables.length];
        String[] parsTrshld = new String[bounds.length + variables.length];
        System.arraycopy(bounds, 0, pars, 0, bounds.length);
        System.arraycopy(variables, 0, pars, bounds.length, variables.length);
        System.arraycopy(bounds, 0, parsTrshld, 0, bounds.length);
        System.arraycopy(trshld, 0, parsTrshld, bounds.length, variables.length);
        double[] timeBoundsLb = Arrays.stream(bounds).mapToDouble(x -> GeneticOptions.min_time_bound).toArray();
        double[] timeBoundsUb = Arrays.stream(bounds).mapToDouble(x -> GeneticOptions.max_time_bound).toArray();
        double[] lbThrshld = Arrays.stream(variables).mapToDouble(popgen::getLowerBound).toArray();
        double[] ubThrshld = Arrays.stream(variables).mapToDouble(popgen::getUpperBound).toArray();
        double[] lb = new double[pars.length];
        double[] ub = new double[pars.length];
        System.arraycopy(timeBoundsLb, 0, lb, 0, timeBoundsLb.length);
        System.arraycopy(lbThrshld, 0, lb, bounds.length, lbThrshld.length);
        System.arraycopy(timeBoundsUb, 0, ub, 0, timeBoundsUb.length);
        System.arraycopy(ubThrshld, 0, ub, bounds.length, ubThrshld.length);
        //double[] start = IntStream.range(0, lb.length).mapToDouble(i -> (lb[i] + ub[i]) / 2).toArray();
        //double[] start = normalize(popgen.getParameterValues(formula.getParameters()), lb, ub);
        ObjectiveFunction function = new FindBestParamtersFromData(popgen, positiveDataset, negativeDataset, times, formula, parsTrshld, timeBounds);
        GPOptimisation gpo = new GPOptimisation();
        GpoOptions options = new GpoOptions();
        //options.setMaxIterations(300);
        options.setHyperparamOptimisation(true);
        KernelRBF kernelGP = new KernelRBF();
        kernelGP.setHyperarameters(new double[]{Math.log(positiveDataset.length + 1), 0.4});
        options.setKernelGP(kernelGP);
        options.setGridSampleNumber(1);
        gpo.setOptions(options);
        GpoResult optimise = gpo.optimise(function, lb, ub);

        System.out.println(optimise);
        return optimise.getSolution();


//        double[] x = start;
//        double rhobeg = 0.5;
//        double rhoend = 1E-3;
//        int iprint = 1;
//        int maxfun = 3500;
//        CobylaExitStatus result = Cobyla.FindMinimum(calcfc, 3, 5, x, rhobeg, rhoend, iprint, maxfun);
//        double[] bestPar = denormalize(x, lb, ub);
//        bestPar = reconvertTimeBounds(pars, bestPar, timeBounds);
//        popgen.setParameters(parsTrshld, bestPar);

    }

    static double[] continuousOptimizationRobustness(FormulaPopulation popgen,String[] variablesUnique, Formula formula, double[][] positiveDataset, double[][] negativeDataset, double[] times) {
        double[] timeBounds = new double[]{GeneticOptions.min_time_bound, GeneticOptions.max_time_bound};
        String[] bounds = formula.getTimeBounds();
        String[] variables = formula.getVariables();
        String[] trshld = formula.getThresholds();
        String[] pars = new String[bounds.length + variables.length];
        String[] parsTrshld = new String[bounds.length + variables.length];
        System.arraycopy(bounds, 0, pars, 0, bounds.length);
        System.arraycopy(variables, 0, pars, bounds.length, variables.length);
        System.arraycopy(bounds, 0, parsTrshld, 0, bounds.length);
        System.arraycopy(trshld, 0, parsTrshld, bounds.length, variables.length);
        double[] timeBoundsLb = Arrays.stream(bounds).mapToDouble(x -> GeneticOptions.min_time_bound).toArray();
        double[] timeBoundsUb = Arrays.stream(bounds).mapToDouble(x -> GeneticOptions.max_time_bound).toArray();
        double[] lbThrshld = Arrays.stream(variables).mapToDouble(popgen::getLowerBound).toArray();
        double[] ubThrshld = Arrays.stream(variables).mapToDouble(popgen::getUpperBound).toArray();
        double[] lb = new double[pars.length];
        double[] ub = new double[pars.length];
        System.arraycopy(timeBoundsLb, 0, lb, 0, timeBoundsLb.length);
        System.arraycopy(lbThrshld, 0, lb, bounds.length, lbThrshld.length);
        System.arraycopy(timeBoundsUb, 0, ub, 0, timeBoundsUb.length);
        System.arraycopy(ubThrshld, 0, ub, bounds.length, ubThrshld.length);
        //double[] start = IntStream.range(0, lb.length).mapToDouble(i -> (lb[i] + ub[i]) / 2).toArray();
        //double[] start = normalize(popgen.getParameterValues(formula.getParameters()), lb, ub);
        ObjectiveFunction function = new FindBestParamtersFromAverage(popgen,variablesUnique, positiveDataset, negativeDataset, times, formula, parsTrshld, timeBounds);
        GPOptimisation gpo = new GPOptimisation();
        GpoOptions options = new GpoOptions();
        //options.setMaxIterations(90);
        options.setHyperparamOptimisation(true);
        KernelRBF kernelGP = new KernelRBF();
        kernelGP.setHyperarameters(new double[]{Math.log(positiveDataset.length + 1), 0.4});
        options.setKernelGP(kernelGP);
        options.setGridSampleNumber(1);
        gpo.setOptions(options);
        GpoResult optimise;

        optimise= gpo.optimise(function, lb, ub);


        //System.out.println(optimise);
        System.out.println("FIT:" + optimise.getFitness());
        return optimise.getSolution();


//        double[] x = start;
//        double rhobeg = 0.5;
//        double rhoend = 1E-3;
//        int iprint = 1;
//        int maxfun = 3500;
//        CobylaExitStatus result = Cobyla.FindMinimum(calcfc, 3, 5, x, rhobeg, rhoend, iprint, maxfun);
//        double[] bestPar = denormalize(x, lb, ub);
//        bestPar = reconvertTimeBounds(pars, bestPar, timeBounds);
//        popgen.setParameters(parsTrshld, bestPar);

    }


    private static double[] reconvertTimeBounds(String[] parN, double[] newP, double[] timeBounds) {
        double[] nP = newP;
        long n = numberOfTimePars(parN);
        for (int i = 0; i < n; i += 2) {
            nP[i] = Math.max(timeBounds[0], newP[i] - newP[i + 1]);
            nP[i + 1] = Math.min(timeBounds[1], newP[i] + newP[i + 1]);
        }
        return nP;
    }

    private static long numberOfTimePars(String[] parN) {
        return Arrays.stream(parN).filter(x -> x.contains("Tl") || x.contains("Tu")).count();
    }

    static void continuousOptimization2(FormulaPopulation popgen, Formula formula, BreathSimulator simulator1, BreathSimulator simulator2, double Tf, int samples) {
        double[] timeBounds = new double[]{GeneticOptions.min_time_bound, GeneticOptions.max_time_bound};
        String[] bounds = formula.getTimeBounds();
        String[] variables = formula.getVariables();
        String[] trshld = formula.getThresholds();
        String[] pars = new String[bounds.length + variables.length];
        String[] parsTrshld = new String[bounds.length + variables.length];
        System.arraycopy(bounds, 0, pars, 0, bounds.length);
        System.arraycopy(variables, 0, pars, bounds.length, variables.length);
        System.arraycopy(bounds, 0, parsTrshld, 0, bounds.length);
        System.arraycopy(trshld, 0, parsTrshld, bounds.length, variables.length);
        double[] timeBoundsLb = Arrays.stream(bounds).mapToDouble(x -> GeneticOptions.min_time_bound).toArray();
        double[] timeBoundsUb = Arrays.stream(bounds).mapToDouble(x -> GeneticOptions.max_time_bound).toArray();
        double[] lbThrshld = Arrays.stream(variables).mapToDouble(popgen::getLowerBound).toArray();
        double[] ubThrshld = Arrays.stream(variables).mapToDouble(popgen::getUpperBound).toArray();
        double[] lb = new double[pars.length];
        double[] ub = new double[pars.length];
        System.arraycopy(timeBoundsLb, 0, lb, 0, timeBoundsLb.length);
        System.arraycopy(lbThrshld, 0, lb, bounds.length, lbThrshld.length);
        System.arraycopy(timeBoundsUb, 0, ub, 0, timeBoundsUb.length);
        System.arraycopy(ubThrshld, 0, ub, bounds.length, ubThrshld.length);
        //double[] start = IntStream.range(0, lb.length).mapToDouble(i -> (lb[i] + ub[i]) / 2).toArray();
        double[] start = popgen.getParameterValues(formula.getParameters());
        double maxValue = -Double.MAX_VALUE;
        double[] bestNp = Arrays.copyOf(start, start.length);
        for (int i = 0; i < 100; i++) {
            System.out.println(i);
//            double[] ran= Arrays.stream(lb).map(v1 -> v1 + Math.random()).toArray();
//            double[] nP= IntStream.range(0, lb.length).mapToDouble(k -> k%2==1&&k<bounds.length ? ran[k-1]+(ub[k]-ran[k-1])*Math.random():k).toArray();

            double[] par = IntStream.range(0, lb.length).mapToDouble(k -> lb[k] + Math.random() * (ub[k] - lb[k])).toArray();
            double[] nP = reconvertTimeBounds(parsTrshld, par, timeBounds);
            int[] data1 = popgen.modelCheck(simulator1, formula, samples, Tf);
            int[] data2 = popgen.modelCheck(simulator2, formula, samples, Tf);
            double v = fitnessBootstrap(formula, data1, data2);
            if (v > maxValue) {
                maxValue = v;
                bestNp = nP;
                System.out.println(maxValue);
                System.out.println(Arrays.toString(bestNp));
            }
        }
        popgen.setParameters(parsTrshld, bestNp);
    }

    private static double fitnessBootstrap(Formula formula, int[] data1, int[] data2) {
        int N = data1.length;
        double p1 = (double) (Arrays.stream(data1).filter(x -> x == 1).count());
        double p2 = (double) (Arrays.stream(data2).filter(x -> x == 1).count());
        double u1 = (double) Arrays.stream(data1).filter(x -> x == -1).sum() / (double) (N);
        double u2 = (double) Arrays.stream(data2).filter(x -> x == -1).sum() / (double) (N);
        FitnessFunction fitness = new RegularisedLogOddRatioFitness();
        return fitness.compute(p1, p2, formula.getFormulaSize(), u1, u2, N);
        //  return (p1+p2);
    }

    public static double[] normalize(double[] data, double[] lb, double[] ub) {
        return IntStream.range(0, data.length).mapToDouble(i -> (data[i] - lb[i]) / (ub[i] - lb[i])).toArray();
    }

    public static double[] denormalize(double[] data, double[] lb, double[] ub) {
        return IntStream.range(0, data.length).mapToDouble(i -> data[i] * (ub[i] - lb[i]) + lb[i]).toArray();
    }

    static double[] smcAverage(FormulaPopulation popgen, Formula formula, String[] variables, double[] times, double[][] trajectories, double[] paramters) {
        return computeAverageRobustness(times,trajectories,variables,formula,paramters);

    }

    public static  double[] computeAverageRobustness(double[] times,double[][] simulate,String[] variables, Formula formula, double[] vv) {
        double[] b = new double[simulate.length];
        Context ns = new Context();
        for (String s : variables) {
            new Variable(s, ns);
        }
        String[] parameters = formula.getParameters();
        StringBuilder builder = new StringBuilder();
        for (int j = 0; j < parameters.length; j++) {
            builder.append("const double ").append(parameters[j]).append("=").append(vv[j]).append(";\n");
        }
        builder.append(formula.toString() + "\n");
        MitlFactory factory = new MitlFactory(ns);
        String text = builder.toString();
        //System.out.println(text);
        MitlPropertiesList l = factory.constructProperties(text);
        MiTL prop = l.getProperties().get(0);

        for (int i = 0; i < simulate.length; i++) {
            Trajectory x = new Trajectory(times, ns, new double[][]{simulate[i]});
            b[i] = prop.evaluateValue(x, 0);
        }
        double mean = Arrays.stream(b).sum()/b.length;
        double variance =  Arrays.stream(b).map(x-> (x-mean)*(x-mean)).sum()/b.length;
        return new double[]{mean,variance};

    }

}
