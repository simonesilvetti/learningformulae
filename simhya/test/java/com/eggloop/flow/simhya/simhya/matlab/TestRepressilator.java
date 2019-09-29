package com.eggloop.flow.simhya.simhya.matlab;


import com.eggloop.flow.numeric.optimization.ObjectiveFunction;
import com.eggloop.flow.numeric.optimization.PointValue;
import com.eggloop.flow.numeric.optimization.methods.PowellMethodApacheConstrained;
import com.eggloop.flow.simhya.simhya.matlab.genetic.*;

import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;
import java.util.Random;
import java.util.stream.IntStream;

public class TestRepressilator {
    public static void main(String[] args) {
        GeneticOptions.setMin_time_bound(0);
        GeneticOptions.setMax_time_bound(1);
        GeneticOptions.setUndefined_reference_threshold(0.1);
        GeneticOptions.setSize_penalty_coefficient(1);
        GeneticOptions.setFitness_type("regularised_logodds");
        GeneticOptions.init__random_number_of_atoms=false;
        GeneticOptions.init__fixed_number_of_atoms=2;

        //TODO: SAGGIUSTARE LA SCALE DA MATLAB
        int N = 10;
        int runs = 1000;
        int Tf = 100;
        FormulaPopulation pop = new FormulaPopulation(N);

//        fitnessOptions.type = 1; %0=normal, 1=modified
//        fitnessOptions.urf = GeneticOptions.undefined_reference_threshold;
//        fitnessOptions.spc = GeneticOptions.size_penalty_coefficient;
//        fitnessOptions.scale = 10;

        int nvar = 3;
        String[] variables = new String[]{"A", "B","C"};
        double[] lower = new double[]{0, 0,0};
        double[] upper = new double[]{1, 1,1};
        for (int i = 0; i < nvar; i++) {
            pop.addVariable(variables[i], lower[i], upper[i]);
        }
        pop.generateInitialPopulation();
        for (int i = 0; i < N; i++) {
            System.out.println(pop.getFormula(i).toString());
        }

//        NormalBreathSimulator normal_model = new NormalBreathSimulator(Test.class.getClassLoader().getResource("data/normal_model/").getPath());
//        normal_model.setFinalExpirationPhaseOnly(true);
//        normal_model.setSavePhase(false);
//        IneffectiveBreathSimulator ineffective_model = new IneffectiveBreathSimulator(Test.class.getClassLoader().getResource("data/ineffective_model/").getPath());
//        ineffective_model.setFinalExpirationPhaseOnly(true);
//        ineffective_model.setSavePhase(false);

        Repressilator normal_model = new Repressilator(0.2, 2, 5, 200);
        Repressilator ineffective_model = new Repressilator(0.2, 6, 8, 200);


        GeneticOptions.setMutate__one_node(false);
        GeneticOptions.setMutate__mutation_probability_per_node(0.1);
        GeneticOptions.setMutate__mutation_probability_one_node(0.5);
        double crossoverProbability = 0.9;
        double migrationProbability = -1.0;
        int numParents = N;
        int termination = 3;
        double terminationIndex = 0;
        int iterationIndex = 0;
        FitnessFunction fitness = new RegularisedLogOddRatioFitness();

        double[][][] p1u1 = new double[50][N][2];
        double[][][] p2u2 = new double[50][N][2];
        double[][] fitnessScore = new double[50][N];
        for (int i = 0; i < N; i++) {
            //     continuousOptimization(pop, pop.getFormula(i), normal_model, ineffective_model, Tf,runs);
            p1u1[iterationIndex][i] = smc(pop, pop.getFormula(i), ineffective_model, Tf, runs);
            p2u2[iterationIndex][i] = smc(pop, pop.getFormula(i), normal_model, Tf, runs);
            fitnessScore[iterationIndex][i] = fitness.compute(p1u1[iterationIndex][i][0], p2u2[iterationIndex][i][0], pop.getFormula(i).getFormulaSize(), p1u1[iterationIndex][i][1], p2u2[iterationIndex][i][1], runs);
        }

        System.out.println(Arrays.toString(fitnessScore[iterationIndex]));
//        double[] fitnessScoreOrdered = Arrays.stream(fitnessScore[iterationIndex]).map(x -> -x).sorted().map(x -> -x).toArray();
        int[] fitnessOrderedAscend = sortGetIndex(fitnessScore[iterationIndex]);
        int[] fitnessOrdered = reverse(fitnessOrderedAscend);
        //int[] fitnessOrdered = Arrays.stream(fitnessOrderedAscend).map(x -> -x).sorted().map(x -> -x).toArray();
        System.out.println(Arrays.toString(fitnessOrdered));

        Formula bformula = pop.getFormula(fitnessOrdered[0]);
        String[] timePar = bformula.getTimeBounds();
        String[] thrsPar = bformula.getThresholds();
        //if (timePar.length + thrsPar.length < 11) {
        continuousOptimization(pop, pop.getFormula(fitnessOrdered[0]), ineffective_model,normal_model,  Tf,runs);
        p1u1[iterationIndex][fitnessOrdered[0]] = smc(pop, pop.getFormula(fitnessOrdered[0]), ineffective_model, Tf, runs);
        p2u2[iterationIndex][fitnessOrdered[0]] = smc(pop, pop.getFormula(fitnessOrdered[0]), normal_model, Tf, runs);
        fitnessScore[iterationIndex][fitnessOrdered[0]] = fitness.compute(p1u1[iterationIndex][fitnessOrdered[0]][0], p2u2[iterationIndex][fitnessOrdered[0]][0], pop.getFormula(fitnessOrdered[0]).getFormulaSize(), p1u1[iterationIndex][fitnessOrdered[0]][1], p2u2[iterationIndex][fitnessOrdered[0]][1], runs);
        //}

// System.out.println(Arrays.toString(fitnessScore[iterationIndex]));
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
                double[] p1u1d = smc(pop, pop.getFormulaNewGeneration(k), ineffective_model, Tf, runs);
                double[] p2u2d = smc(pop, pop.getFormulaNewGeneration(k), normal_model, Tf, runs);
                double fitnessDescendant1 = fitness.compute(p1u1d[0], p2u2d[0], pop.getFormulaNewGeneration(k).getFormulaSize(), p1u1d[1], p2u2d[1], runs);
                p1u1d = smc(pop, pop.getFormulaNewGeneration(k + 1), ineffective_model, Tf, runs);
                p2u2d = smc(pop, pop.getFormulaNewGeneration(k + 1), normal_model, Tf, runs);
                double fitnessDescendant2 = fitness.compute(p1u1d[0], p2u2d[0], pop.getFormulaNewGeneration(k + 1).getFormulaSize(), p1u1d[1], p2u2d[1], runs);
                if (fitnessDescendant1 < fitnessDescendant2) {
                    pop.removeFormulaNewPopulation(k);
                } else {
                    pop.removeFormulaNewPopulation(k + 1);
                }
                pop.mutateNewGeneration(k);
                if (pop.getFormulaNewGeneration(k) != parent1 && pop.getFormulaNewGeneration(k) != parent2) {
                    //continuousOptimization(pop, pop.getFormula(k), normal_model, ineffective_model, Tf,runs);
                    p1u1[iterationIndex + 1][k] = smc(pop, pop.getFormula(k), ineffective_model, Tf, runs);
                    p2u2[iterationIndex + 1][k] = smc(pop, pop.getFormula(k), normal_model, Tf, runs);
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

            //fitnessOrderedAscend = sortGetIndex(fitnessScore[iterationIndex + 1]);
            //fitnessOrdered = Arrays.stream(fitnessOrderedAscend).map(x -> -x).sorted().map(x -> -x).toArray();
            double migrationIndex = Math.random();
            if (migrationIndex <= migrationProbability) {
                pop.removeFormulaNewPopulation(fitnessOrdered[fitnessOrdered.length - 1]);
                int newIndividualIndex = pop.addRandomFormula();
                continuousOptimization(pop, pop.getFormula(newIndividualIndex), ineffective_model,normal_model,  Tf,runs);
                p1u1[iterationIndex + 1][newIndividualIndex] = smc(pop, pop.getFormula(newIndividualIndex), ineffective_model, Tf, runs);
                p2u2[iterationIndex + 1][newIndividualIndex ] = smc(pop, pop.getFormula(newIndividualIndex), normal_model, Tf, runs);
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
        System.out.println(bestFormula.toString());


        double p1bestFormula = p1u1[iterationIndex][fitnessOrdered[0]][0];
        double p2bestFormula = p2u2[iterationIndex][fitnessOrdered[0]][0];
        double u1bestFormula = p1u1[iterationIndex][fitnessOrdered[0]][1];
        double u2bestFormula = p2u2[iterationIndex][fitnessOrdered[0]][1];
        System.out.println("p1bestFormula= " + p1bestFormula);
        System.out.println("p2bestFormula= " + p2bestFormula);
        System.out.println("u1bestFormula= " + u1bestFormula);
        System.out.println("u2bestFormula= " + u2bestFormula);

        System.out.println("ineffective= " + smc(pop, bestFormula, ineffective_model, Tf, runs)[0]);
        System.out.println("normal= " + smc(pop, bestFormula, normal_model, Tf, runs)[0]);
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

    static double[] smc(FormulaPopulation popgen, Formula formula, BasicSimulator simulator, double Tf, int samples) {
        int[] data = popgen.modelCheck(simulator, formula, samples, Tf);
        double p = (double) (Arrays.stream(data).filter(x -> x == 1).count() + 1) / (samples + 2);
        double u = (double) (Arrays.stream(data).filter(x -> x == -1).count()) / (samples);
//        p = (sum(data==1)+1)/(samples+2);
//        u = (sum(data==-1))/(samples)
        return new double[]{p, u};

    }

    static void continuousOptimization(FormulaPopulation popgen, Formula formula, BasicSimulator simulator1, BasicSimulator simulator2, double Tf, int samples) {
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
        PowellMethodApacheConstrained alg = new PowellMethodApacheConstrained();
        ObjectiveFunction function = new FindBestParamtersFromSimulations(popgen, simulator1, simulator2, Tf, formula, parsTrshld, samples, timeBounds);
        PointValue best = alg.optimise(function, start,lb,ub);
        double[] bestPar = best.getPoint();
        bestPar = reconvertTimeBounds(pars,bestPar,timeBounds);
        popgen.setParameters(parsTrshld,bestPar);
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
        double[] bestNp = Arrays.copyOf(start,start.length);
        for (int i = 0; i < 50; i++) {
            double[] par = IntStream.range(0, lb.length).mapToDouble(k -> lb[k] + Math.random() * (ub[k] - lb[k])).toArray();
            double[] nP = reconvertTimeBounds(parsTrshld, par, timeBounds);
            int[] data1 = popgen.modelCheck(simulator1, formula, samples, Tf);
            int[] data2 = popgen.modelCheck(simulator2, formula, samples, Tf);
            double v = fitnessBootstrap(popgen, formula, data1, data2);
            if (v > maxValue) {
                maxValue = v;
                bestNp = nP;
            }
        }
        popgen.setParameters(parsTrshld,bestNp);
    }

    static double OptimisationFunction(FormulaPopulation popgen, BreathSimulator simfunc1, BreathSimulator simfunc2, double Tf, Formula formula, String[] parN, double[] newP, int samples, double[] timeBounds) {
        double[] nP = reconvertTimeBounds(parN, newP, timeBounds);
        popgen.setParameters(parN, nP);
        int[] data1 = popgen.modelCheck(simfunc1, formula, samples, Tf);
        int[] data2 = popgen.modelCheck(simfunc2, formula, samples, Tf);
        return fitnessBootstrap(popgen, formula, data1, data2);
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

    private static double fitnessBootstrap(FormulaPopulation popgen, Formula formula, int[] data1, int[] data2) {
        int N = data1.length;
        double p1 = (Arrays.stream(data1).filter(x -> x == 1).sum() + 1) / (N + 2);
        double p2 = (Arrays.stream(data2).filter(x -> x == 1).sum() + 1) / (N + 2);
        double u1 = Arrays.stream(data1).filter(x -> x == -1).sum() / (N);
        double u2 = Arrays.stream(data2).filter(x -> x == -1).sum() / (N);
        FitnessFunction fitness = new RegularisedLogOddRatioFitness();
        return fitness.compute(p1, p2, formula.getFormulaSize(), u1, u2, N);
    }


    static double[] averageRobustness(FormulaPopulation popgen, Formula formula, BreathSimulator simulator,
                                      double Tf, int samples) {
        int[] data = popgen.modelCheck(simulator, formula, samples, Tf);
        double p = (double) (Arrays.stream(data).filter(x -> x == 1).count() + 1) / (samples + 2);
        double u = (double) (Arrays.stream(data).filter(x -> x == -1).count()) / (samples);
//        p = (sum(data==1)+1)/(samples+2);
//        u = (sum(data==-1))/(samples)
        return new double[]{p, u};

    }

    public static int[] reverse(int[] nums) {
        int[] reversed = new int[nums.length];
        for (int i = 0; i < nums.length; i++) {
            reversed[i] = nums[nums.length - 1 - i];
        }
        return reversed;
    }

}

