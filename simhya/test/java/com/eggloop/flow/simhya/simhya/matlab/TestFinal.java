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
import com.eggloop.flow.simhya.simhya.modelchecking.mtl.MTLformula;
import com.eggloop.flow.utils.data.TrajectoryReconstruction;
import com.eggloop.flow.utils.files.Utils;

import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;
import java.util.Random;
import java.util.stream.IntStream;

public class TestFinal {
    public static void main(String[] args) {
        Random ran = new Random(1);
        double[][] ds2SpatialValues = Utils.readMatrixFromFile(TestReading.class.getClassLoader().getResource("data/calin/ds2Trajectories").getPath());
        double[] ds2Labels = Utils.readVectorFromFile(TestReading.class.getClassLoader().getResource("data/calin/ds2Labels").getPath());
        double[] ds2Times = Utils.readVectorFromFile(TestReading.class.getClassLoader().getResource("data/calin/ds2Times").getPath());
        GeneticOptions.setMin_time_bound(0);
        GeneticOptions.setMax_time_bound(100);
        GeneticOptions.setUndefined_reference_threshold(0.1);
        GeneticOptions.setSize_penalty_coefficient(1);
        GeneticOptions.setInit__average_number_of_atoms(3);
        //GeneticOptions.setInit__fixed_number_of_atoms(2);
        GeneticOptions.setFitness_type("regularised_logodds");
        GeneticOptions.init__random_number_of_atoms=true;
        //GeneticOptions.init__fixed_number_of_atoms=2;
        GeneticOptions.setSize_penalty_coefficient(1);

        //TODO: SAGGIUSTARE LA SCALE DA MATLAB
        int N = 3;
        int runs = 1000;
        int Tf = 100;
        FormulaPopulation pop  = new FormulaPopulation(N);

//        fitnessOptions.type = 1; %0=normal, 1=modified
//        fitnessOptions.urf = GeneticOptions.undefined_reference_threshold;
//        fitnessOptions.spc = GeneticOptions.size_penalty_coefficient;
//        fitnessOptions.scale = 10;


        String[] variables = new String[]{"flow"};
        double[] lower = new double[]{0};
        double[] upper = new double[]{12};
        for (int i = 0; i < variables.length; i++) {
            pop.addVariable(variables[i], lower[i], upper[i]);
        }
        pop.generateInitialPopulation();
        for (int i = 0; i < N; i++) {
            System.out.println(pop.getFormula(i).toString());
        }

        TrajectoryReconstruction data = new TrajectoryReconstruction(ds2Times, ds2SpatialValues, ds2Labels, 0.7,ran);
        data.split();
        double[][] normal_model = data.getPoistiveTrainSet();
        double[][] ineffective_model = data.getNegativeTrainSet();
        double[][] normal_model_test = data.getPoistiveValidationSet();
        double[][] ineffective_model_test = data.getNegativeValidationSet();

 //       computeAverageRobustness(ds2Times,normal_model,pop.getFormula(0),pop.getParameterValues(pop.getFormula(0).getParameters()));

        GeneticOptions.setMutate__one_node(false);
        //GeneticOptions.setMutate__mutation_probability_per_node(0.1);
        //GeneticOptions.setMutate__mutation_probability_one_node(0.5);
        double crossoverProbability = 0.9;
        double migrationProbability =0.0;
        int numParents = N;
        int termination = 3;
        double terminationIndex = 0;
        int iterationIndex = 0;
        FitnessFunction fitness = new NewFitness();
//
//        String formulazza ="const double Tl_2=29.00556116178467;\n" +
//                "const double Tu_2=82.95934229254344;\n" +
//                "const double Tl_3=0.0;\n" +
//                "const double Tu_3=59.73873157847135;\n" +
//                "const double Tl_5=37.27425053848329;\n" +
//                "const double Tu_5=37.43831321145746;\n" +
//                "const double Theta_0=8.274101074253807;\n" +
//                "const double Theta_1=8.5649684343209;\n" +
//                "F[Tl_5, Tu_5] ((flow <= Theta_0 & G[Tl_3, Tu_3] (F[Tl_2, Tu_2] (flow <= Theta_1))))\n";
//        String formulazza2 = "const double Tl_1=37.27425053848329;\n" +
//                "const double Tu_1=37.43831321145746;\n" +
//                "const double Tl_2=0.0;\n" +
//                "const double Tu_2=71.37473695267416;\n" +
//                "const double Theta_0=0.8309585389109331;\n" +
//                "G[Tl_2, Tu_2] (F[Tl_1, Tu_1] (flow >= Theta_0))\n";
//        String[] vvv = new String[]{"flow"};
//        Context ns = new Context();
//        for (String s : vvv) {
//            new Variable(s, ns);
//        }
//
//        for (int i = 0; i < normal_model.length; i++) {
//            Trajectory x = new Trajectory(ds2Times, ns, new double[][]{normal_model[i]});
//            MitlFactory factory = new MitlFactory(ns);
//
//            MitlPropertiesList l = factory.constructProperties(formulazza2);
//            MiTL prop = l.getProperties().get(0);
//            prop.evaluateValue(x, 0);
//        }







        double[][][] p1u1 = new double[50][N][2];
        double[][][] p2u2 = new double[50][N][2];
        double[][] fitnessScore = new double[50][N];
        for (int i = 0; i < N; i++) {
            continuousOptimization(pop,variables, pop.getFormula(i), normal_model, ineffective_model, ds2Times);
            p1u1[iterationIndex][i] = smcAverage(pop, pop.getFormula(i),variables,ds2Times, ineffective_model);
            p2u2[iterationIndex][i] = smcAverage(pop, pop.getFormula(i),variables,ds2Times, normal_model);
            fitnessScore[iterationIndex][i] = fitness.compute(p1u1[iterationIndex][i][0], p2u2[iterationIndex][i][0], pop.getFormula(i).getFormulaSize(), p1u1[iterationIndex][i][1], p2u2[iterationIndex][i][1], runs);
        }
        int[] fitnessOrderedAscend = sortGetIndex(fitnessScore[iterationIndex]);
        int[] fitnessOrdered = reverse(fitnessOrderedAscend);
        System.out.println(Arrays.toString(fitnessScore[iterationIndex]));
        Formula bformula = pop.getFormula(fitnessOrdered[0]);
        String[] timePar = bformula.getTimeBounds();
        String[] thrsPar = bformula.getThresholds();
        if (timePar.length + thrsPar.length < 11) {
            continuousOptimization(pop,variables, bformula, ineffective_model,normal_model,ds2Times);
            p1u1[iterationIndex][fitnessOrdered[0]] = smc(pop, bformula,ds2Times, ineffective_model);
            p2u2[iterationIndex][fitnessOrdered[0]] = smc(pop, bformula,ds2Times, normal_model);
            fitnessScore[iterationIndex][fitnessOrdered[0]] = fitness.compute(p1u1[iterationIndex][fitnessOrdered[0]][0], p2u2[iterationIndex][fitnessOrdered[0]][0], pop.getFormula(fitnessOrdered[0]).getFormulaSize(), p1u1[iterationIndex][fitnessOrdered[0]][1], p2u2[iterationIndex][fitnessOrdered[0]][1], runs);
        }

        double bestFormulaFitness = fitnessScore[iterationIndex][fitnessOrdered[0]];
        int[][] parentalSet = truncationSelection(fitnessOrdered, numParents, 8);
        System.out.println(Arrays.toString(parentalSet[0]));
        while (terminationIndex < termination) {
            pop.initialiseNewGeneration();
            fitnessOrderedAscend = sortGetIndex(fitnessScore[iterationIndex]);
            fitnessOrdered = reverse(fitnessOrderedAscend);
            parentalSet = truncationSelection(fitnessOrdered, numParents, 8);
            for (int k = 0; k < numParents; k++) {
                pop.selectFormula(parentalSet[k][0]);
                pop.selectFormula(parentalSet[k][1]);
                Formula parent1 = pop.getFormulaNewGeneration(k);
                Formula parent2 = pop.getFormulaNewGeneration(k + 1);
                double crossoverIndex = Math.random();
                if (crossoverIndex <= crossoverProbability) {
                    String[] c11 = parent1.getTimeBounds();
                    String[] c12 = parent1.getThresholds();
                    String[] c21 = parent2.getTimeBounds();
                    String[] c22 = parent2.getThresholds();
                    if (c11.length + c12.length > 1 && c21.length + c22.length > 1) {
                        pop.crossoverNewGeneration(k, k + 1);
                    }
                }
                double[] p1u1d = smc(pop, pop.getFormulaNewGeneration(k),ds2Times, ineffective_model);
                double[] p2u2d = smc(pop, pop.getFormulaNewGeneration(k),ds2Times, normal_model);
                double fitnessDescendant1 = fitness.compute(p1u1d[0], p2u2d[0], pop.getFormulaNewGeneration(k).getFormulaSize(), p1u1d[1], p2u2d[1], runs);
                p1u1d = smc(pop, pop.getFormulaNewGeneration(k + 1),ds2Times, ineffective_model);
                p2u2d = smc(pop, pop.getFormulaNewGeneration(k + 1),ds2Times, normal_model);
                double fitnessDescendant2 = fitness.compute(p1u1d[0], p2u2d[0], pop.getFormulaNewGeneration(k + 1).getFormulaSize(), p1u1d[1], p2u2d[1], runs);
                if (fitnessDescendant1 < fitnessDescendant2) {
                    pop.removeFormulaNewPopulation(k);
                } else {
                    pop.removeFormulaNewPopulation(k + 1);
                }
                pop.mutateNewGeneration(k);
                if (pop.getFormulaNewGeneration(k) != parent1 && pop.getFormulaNewGeneration(k) != parent2) {
                    continuousOptimization(pop,variables, pop.getFormula(k), normal_model, ineffective_model,ds2Times);
                    p1u1[iterationIndex + 1][k] = smc(pop, pop.getFormula(k), ds2Times,ineffective_model);
                    p2u2[iterationIndex + 1][k] = smc(pop, pop.getFormula(k), ds2Times,normal_model);
                    fitnessScore[iterationIndex + 1][k] = fitness.compute(p1u1[iterationIndex + 1][k][0], p2u2[iterationIndex + 1][k][0], pop.getFormula(k + 1).getFormulaSize(), p1u1[iterationIndex][k][1], p2u2[iterationIndex][k][1], runs);
                } else if (pop.getFormulaNewGeneration(k) == parent1) {
                    p1u1[iterationIndex + 1][k] = p1u1[iterationIndex][parentalSet[k][0]];
                    p2u2[iterationIndex + 1][k] = p2u2[iterationIndex][parentalSet[k][0]];
                    fitnessScore[iterationIndex + 1][k] = fitnessScore[iterationIndex][parentalSet[k][0]];
                } else {
                    p1u1[iterationIndex + 1][k] = p1u1[iterationIndex][parentalSet[k][1]];
                    p2u2[iterationIndex + 1][k] = p2u2[iterationIndex][parentalSet[k][1]];
                    fitnessScore[iterationIndex + 1][k] = fitnessScore[iterationIndex][parentalSet[k][1]];
                }
            }
            for (int k = 0; k < N - numParents - 1; k++) {
                p1u1[iterationIndex + 1][numParents + k] = p1u1[iterationIndex][fitnessOrdered[k]];
                p2u2[iterationIndex + 1][numParents + k] = p2u2[iterationIndex][fitnessOrdered[k]];
                fitnessScore[iterationIndex + 1][numParents + k] = fitnessScore[iterationIndex][fitnessOrdered[k]];

            }

            fitnessOrderedAscend = sortGetIndex(fitnessScore[iterationIndex+1]);
            fitnessOrdered = reverse(fitnessOrderedAscend);
//            if (timePar.length + thrsPar.length < 11) {
//                continuousOptimization(pop, pop.getFormula(fitnessOrdered[0]), ineffective_model,normal_model,  Tf,runs);
//                p1u1[iterationIndex+1][fitnessOrdered[0]] = smc(pop, pop.getFormula(fitnessOrdered[0]), ineffective_model, Tf, runs);
//                p2u2[iterationIndex+1][fitnessOrdered[0]] = smc(pop, pop.getFormula(fitnessOrdered[0]), normal_model, Tf, runs);
//                fitnessScore[iterationIndex+1][fitnessOrdered[0]] = fitness.compute(p1u1[iterationIndex+1][fitnessOrdered[0]][0], p2u2[iterationIndex+1][fitnessOrdered[0]][0], pop.getFormula(fitnessOrdered[0]).getFormulaSize(), p1u1[iterationIndex+1][fitnessOrdered[0]][1], p2u2[iterationIndex+1][fitnessOrdered[0]][1], runs);
//            }


            //fitnessOrderedAscend = sortGetIndex(fitnessScore[iterationIndex + 1]);
            //fitnessOrdered = Arrays.stream(fitnessOrderedAscend).map(x -> -x).sorted().map(x -> -x).toArray();
            double migrationIndex = Math.random();
            if (migrationIndex <= migrationProbability) {
                pop.removeFormulaNewPopulation(fitnessOrdered[fitnessOrdered.length - 1]);
                int newIndividualIndex = pop.addRandomFormula();
                continuousOptimization(pop,variables, pop.getFormula(newIndividualIndex), ineffective_model,normal_model, ds2Times);
                p1u1[iterationIndex + 1][newIndividualIndex] = smc(pop, pop.getFormula(newIndividualIndex), ds2Times, ineffective_model_test);
                p2u2[iterationIndex + 1][newIndividualIndex ] = smc(pop, pop.getFormula(newIndividualIndex), ds2Times, normal_model_test);
                fitnessScore[iterationIndex + 1][newIndividualIndex] = fitness.compute(p1u1[iterationIndex + 1][newIndividualIndex][0], p2u2[iterationIndex + 1][newIndividualIndex][0], pop.getFormula(newIndividualIndex).getFormulaSize(), p1u1[iterationIndex][newIndividualIndex][1], p2u2[iterationIndex][newIndividualIndex][1], runs);
                //fitnessOrderedAscend = sortGetIndex(fitnessScore[iterationIndex + 1]);
                //fitnessOrdered = Arrays.stream(fitnessOrderedAscend).map(x -> -x).sorted().map(x -> -x).toArray();
                fitnessOrderedAscend = sortGetIndex(fitnessScore[iterationIndex]);
                fitnessOrdered = reverse(fitnessOrderedAscend);
            }
            double bestNewFormulaFitness = fitnessScore[iterationIndex + 1][fitnessOrdered[1]];
            if (bestFormulaFitness == bestNewFormulaFitness) {
                terminationIndex = terminationIndex + 1;
            } else {
                terminationIndex = 0;
                bestFormulaFitness = bestNewFormulaFitness;
            }
            System.out.printf("TI: " + terminationIndex);
            pop.finaliseNewGeneration();
            iterationIndex = iterationIndex + 1;
            System.out.println();
        }
        Formula bestFormula = pop.getFormula(fitnessOrdered[0]);



        double p1bestFormula = p1u1[iterationIndex][fitnessOrdered[0]][0];
        double p2bestFormula = p2u2[iterationIndex][fitnessOrdered[0]][0];
        double u1bestFormula = p1u1[iterationIndex][fitnessOrdered[0]][1];
        double u2bestFormula = p2u2[iterationIndex][fitnessOrdered[0]][1];
        System.out.println("p1bestFormula= " + p1bestFormula);
        System.out.println("p2bestFormula= " + p2bestFormula);
        System.out.println("u1bestFormula= " + u1bestFormula);
        System.out.println("u2bestFormula= " + u2bestFormula);
        double[] paramters = continuousOptimization(pop,variables, bestFormula, ineffective_model,normal_model, ds2Times);
        pop.setParameters(bestFormula.getParameters(),paramters);
        System.out.println("ineffective= " + smc(pop, bestFormula, ds2Times, ineffective_model_test)[0]);
        System.out.println("normal= " + smc(pop, bestFormula, ds2Times, normal_model_test)[0]);
        System.out.println(bestFormula.toString());
        System.out.println(Arrays.toString(paramters));

    }


    static double[] smc(FormulaPopulation popgen, Formula formula, double[] times, double[][] trajectories) {
        int[] data = new int[trajectories.length];
        for (int i = 0; i < trajectories.length; i++) {
            data[i] = popgen.modelCheck(combineTimeAndSpace(times, new double[][]{trajectories[i]}), formula);
        }
        double p = (double) (Arrays.stream(data).filter(x -> x == 1).count() + 1) / (trajectories.length + 2);
        double u = (double) (Arrays.stream(data).filter(x -> x == -1).count()) / (trajectories.length);
//        p = (sum(data==1)+1)/(samples+2);
//        u = (sum(data==-1))/(samples)
        return new double[]{p, u};

    }

    static double[] smcAverage(FormulaPopulation popgen, Formula formula, String[] variables, double[] times, double[][] trajectories) {
       return computeAverageRobustness(times,trajectories,variables,formula,popgen.getParameterValues(formula.getParameters()));

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


    static double[] continuousOptimization(FormulaPopulation popgen, String[] variablesUnique, Formula formula, double[][] positiveDataset, double[][] negativeDataset, double[] times) {
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
        options.setHyperparamOptimisation(false);
        KernelRBF kernelGP = new KernelRBF();
        kernelGP.setHyperarameters(new double[]{Math.log(positiveDataset.length + 1), 0.4});
        options.setKernelGP(kernelGP);
        options.setGridSampleNumber(1);
        gpo.setOptions(options);
        GpoResult optimise = gpo.optimise(function, lb, ub);

        //System.out.println(optimise);
        double[]bestPar = reconvertTimeBounds(pars,optimise.getSolution(),timeBounds);
        popgen.setParameters(parsTrshld,bestPar);
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

    private static int[] sortGetIndex(double[] values) {
        Map<Double, Integer> indices = new HashMap<>();
        for (int index = 0; index < values.length; index++) {
            indices.put(values[index], index);
        }

        double[] copy = Arrays.copyOf(values, values.length);
        int[] indici = new int[values.length];
        Arrays.sort(copy);
        for (int index = 0; index < copy.length; index++) {
            indici[index] = indices.get(copy[index]);
        }
        return indici;
    }

    private static int[][] truncationSelection(int[] fitnessOrdered, int numParents, int i) {
        int[] candidateParents = Arrays.copyOfRange(fitnessOrdered, 0, i);
        int[][] parental = new int[numParents][];
        for (int j = 0; j < parental.length; j++) {
            parental[j] = new int[]{candidateParents[(int) Math.floor((Math.random() * candidateParents.length))], candidateParents[(int) Math.floor((Math.random() * candidateParents.length))]};
        }
        return parental;
    }


    public static int[] reverse(int[] nums) {
        int[] reversed = new int[nums.length];
        for (int i = 0; i < nums.length; i++) {
            reversed[i] = nums[nums.length - 1 - i];
        }
        return reversed;
    }

    public static  double[] computeAverageRobustness(double[] times,double[][] simulate,String[] variables, Formula formula, double[] vv) {
        double[] b = new double[simulate.length];

        for (int i = 0; i < simulate.length; i++) {
            Context ns = new Context();
            for (String s : variables) {
                new Variable(s, ns);
            }
            Trajectory x = new Trajectory(times, ns, new double[][]{simulate[i]});
            String[] parameters = formula.getParameters();
            StringBuilder builder = new StringBuilder();
            for (int j = 0; j < parameters.length; j++) {
                builder.append("const double ").append(parameters[j]).append("=").append(vv[j]).append("\n");
            }
            builder.append(formula.toString() + "\n");
            MitlFactory factory = new MitlFactory(ns);
            String text = builder.toString();
            //System.out.println(text);
            MitlPropertiesList l = factory.constructProperties(text);
            MiTL prop = l.getProperties().get(0);
            b[i] = prop.evaluateValue(x, 0);
        }
        double mean = Arrays.stream(b).sum()/b.length;
        double variance =  Arrays.stream(b).map(x-> (x-mean)*(x-mean)).sum()/b.length;
        return new double[]{mean,variance};

    }





}
