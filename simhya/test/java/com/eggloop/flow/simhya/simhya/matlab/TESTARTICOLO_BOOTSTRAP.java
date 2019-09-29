package com.eggloop.flow.simhya.simhya.matlab;

import com.eggloop.flow.expr.Context;
import com.eggloop.flow.expr.Variable;
import com.eggloop.flow.mitl.MiTL;
import com.eggloop.flow.mitl.MitlPropertiesList;
import com.eggloop.flow.model.Trajectory;
import com.eggloop.flow.parsers.MitlFactory;
import com.eggloop.flow.simhya.simhya.matlab.genetic.Formula;
import com.eggloop.flow.simhya.simhya.matlab.genetic.FormulaPopulation;
import com.eggloop.flow.simhya.simhya.matlab.genetic.GeneticOptions;
import com.eggloop.flow.simhya.simhya.matlab.genetic.NewFitness;
import com.eggloop.flow.utils.data.Bootstrap;
import com.eggloop.flow.utils.data.TrajectoryReconstruction;
import com.eggloop.flow.utils.files.Utils;

import java.util.*;

public class TESTARTICOLO_BOOTSTRAP {
    public static void main(String[] args) {
        Random ran = new Random(1);
        double[][] ds2SpatialValues = Utils.readMatrixFromFile(TestReading.class.getClassLoader().getResource("data/calin/ds2Trajectories").getPath());
        double[] ds2Labels = Utils.readVectorFromFile(TestReading.class.getClassLoader().getResource("data/calin/ds2Labels").getPath());
        double[] ds2Times = Utils.readVectorFromFile(TestReading.class.getClassLoader().getResource("data/calin/ds2Times").getPath());
        TrajectoryReconstruction data = new TrajectoryReconstruction(ds2Times, ds2SpatialValues, ds2Labels, 0.7, ran);
        data.split();
        double[][] normal_model_pre = data.getPoistiveTrainSet();
        double[][] ineffective_model_pre = data.getNegativeTrainSet();
        double[][] normal_model_test = data.getPoistiveValidationSet();
        double[][] ineffective_model_test = data.getNegativeValidationSet();

        int seed=1;
        int boot=10;
        List<Formula> formulae = new ArrayList<>();
        List<double[]> param = new ArrayList<>();



        int N = 7;

        double crossoverProbability = 0.9;
        double migrationProbability = 0.0;
        int numParents = N;
        int termination = 4;
        double terminationIndex = 0;
        int iterationIndex = 0;
        double best = -Double.POSITIVE_INFINITY;
        Formula bbbFormula = null;
        double[] bbbParamters = null;
        FormulaPopulation pop = new FormulaPopulation(N);
        String[] variables = new String[]{"flow"};
        double[] lower = new double[]{0};
        double[] upper = new double[]{12};
        for (int i = 0; i < variables.length; i++) {
            pop.addVariable(variables[i], lower[i], upper[i]);
        }
        // GeneticOptions.init__globallyeventually_weight=0.2;
        //GeneticOptions.init__eventuallyglobally_weight=0.2;
        //GeneticOptions.eventu
        //GeneticOptions.init__not_weight=0;
        //GeneticOptions.init__until_weight=0.4;
        //GeneticOptions.init
        //GeneticOptions.setInit__fixed_number_of_atoms(4);
        GeneticOptions.setInit__prob_of_true_atom(0);
        GeneticOptions.setMin_time_bound(0);
        GeneticOptions.setMax_time_bound(100);

        for (int kk = 0; kk <boot ; kk++) {
            seed=seed+3;
            ran.setSeed(seed);
            double[][] normal_model = Bootstrap.bootstrap(50, normal_model_pre, ran);
            double[][] ineffective_model = Bootstrap.bootstrap(50, ineffective_model_pre, ran);



            pop.generateInitialPopulation();
            pop.initialiseNewGeneration();

            for (int i = 0; i < pop.getPopulationSize(); i++) {
                //System.out.println(pop.getFormula(i).toString());
            }
            //System.out.println("a");
            NewFitness fitness = new NewFitness();
            double[][][] p1u1 = new double[50][N][];
            double[][][] p2u2 = new double[50][N][];
            double[][] fitnessScore = new double[50][N];
            for (int i = 0; i < N; i++) {
                double[] vv = ComputeAverage.average(ds2Times, normal_model, ineffective_model, pop.getFormula(i), pop, new double[]{GeneticOptions.min_time_bound, GeneticOptions.max_time_bound});
                //  System.out.println(Arrays.toString(vv));
                p1u1[iterationIndex][i] = ComputeAverage.computeAverageRobustness(ds2Times, normal_model, variables, pop.getFormula(i), vv);
                p2u2[iterationIndex][i] = ComputeAverage.computeAverageRobustness(ds2Times, ineffective_model, variables, pop.getFormula(i), vv);
                fitnessScore[iterationIndex][i] = fitness.compute(p1u1[iterationIndex][i][0], p2u2[iterationIndex][i][0], pop.getFormula(i).getFormulaSize(), p1u1[iterationIndex][i][1], p2u2[iterationIndex][i][1], 0);
                if (fitnessScore[iterationIndex][i] > best) {
                    best = fitnessScore[iterationIndex][i];
                    bbbFormula = pop.getFormula(i);
                    bbbParamters = Arrays.copyOf(vv, vv.length);
                }

            }


            //System.out.println("11111111111111111111111");

            int[] fitnessOrderedAscend = sortGetIndex(fitnessScore[iterationIndex]);
            int[] fitnessOrdered = reverse(fitnessOrderedAscend);
            //System.out.println("EEEEEEEEEEE"+Arrays.toString(fitnessScore[iterationIndex]));
            //System.out.println("SSSSSSSSSSS:"+Arrays.toString(fitnessOrdered));
            //System.out.println(Arrays.toString(fitnessScore[iterationIndex]));
            Formula bformula = pop.getFormula(fitnessOrdered[0]);
            String[] timePar = bformula.getTimeBounds();
            String[] thrsPar = bformula.getThresholds();
            if (timePar.length + thrsPar.length < 11) {
                //continuousOptimization(pop,variables, bformula, ineffective_model,normal_model,ds2Times);
                double[] vv = ComputeAverage.average(ds2Times, normal_model, ineffective_model, bformula, pop, new double[]{GeneticOptions.min_time_bound, GeneticOptions.max_time_bound});
                p1u1[iterationIndex][fitnessOrdered[0]] = ComputeAverage.computeAverageRobustness(ds2Times, ineffective_model, variables, bformula, vv);
                p2u2[iterationIndex][fitnessOrdered[0]] = ComputeAverage.computeAverageRobustness(ds2Times, normal_model, variables, bformula, vv);
//            p1u1[iterationIndex][fitnessOrdered[0]] = smc(pop, bformula,ds2Times, ineffective_model);
//            p2u2[iterationIndex][fitnessOrdered[0]] = smc(pop, bformula,ds2Times, normal_model);
                fitnessScore[iterationIndex][fitnessOrdered[0]] = fitness.compute(p1u1[iterationIndex][fitnessOrdered[0]][0], p2u2[iterationIndex][fitnessOrdered[0]][0], pop.getFormula(fitnessOrdered[0]).getFormulaSize(), p1u1[iterationIndex][fitnessOrdered[0]][1], p2u2[iterationIndex][fitnessOrdered[0]][1], 0);
                if (fitnessScore[iterationIndex][fitnessOrdered[0]] > best) {
                    best = fitnessScore[iterationIndex][fitnessOrdered[0]];
                    bbbFormula = pop.getFormula(fitnessOrdered[0]);
                    bbbParamters = Arrays.copyOf(vv, vv.length);
                }
            }

            double bestFormulaFitness = fitnessScore[iterationIndex][fitnessOrdered[0]];
            int[][] parentalSet = truncationSelection(fitnessOrdered, numParents, 8);
            // System.out.println(Arrays.toString(parentalSet[0]));

            // System.out.println("11111111111111111111111");
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
                            // System.out.println("CROSSS");
                        }
                    }
                    // System.out.println(" CCC= "+k);
                    double[] vv = ComputeAverage.average(ds2Times, normal_model, ineffective_model, pop.getFormulaNewGeneration(k), pop, new double[]{GeneticOptions.min_time_bound, GeneticOptions.max_time_bound});
                    for (int i = 0; i < vv.length; i++) {
                        if (Double.isNaN(vv[i])) {
                            vv[i] = 0;
                        }
                    }
                    // System.out.println("11111111111111111111111");
                    double[] p1u1d = ComputeAverage.computeAverageRobustness(ds2Times, ineffective_model, variables, pop.getFormulaNewGeneration(k), vv);
                    double[] p2u2d = ComputeAverage.computeAverageRobustness(ds2Times, normal_model, variables, pop.getFormulaNewGeneration(k), vv);

//                double[] p1u1d = smc(pop, pop.getFormulaNewGeneration(k),ds2Times, ineffective_model);
//                double[] p2u2d = smc(pop, pop.getFormulaNewGeneration(k),ds2Times, normal_model);
                    double fitnessDescendant1 = fitness.compute(p1u1d[0], p2u2d[0], pop.getFormulaNewGeneration(k).getFormulaSize(), p1u1d[1], p2u2d[1], 0);
                    if (fitnessDescendant1 > best) {
                        best = fitnessDescendant1;
                        bbbFormula = pop.getFormulaNewGeneration(k);
                        bbbParamters = Arrays.copyOf(vv, vv.length);
                    }
                    // System.out.println(" AAA= "+k);
                    vv = ComputeAverage.average(ds2Times, normal_model, ineffective_model, pop.getFormulaNewGeneration(k + 1), pop, new double[]{GeneticOptions.min_time_bound, GeneticOptions.max_time_bound});
                    for (int i = 0; i < vv.length; i++) {
                        if (Double.isNaN(vv[i])) {
                            vv[i] = 0;
                        }
                    }
                    //System.out.println(" BBB= "+ Arrays.toString(vv));
                    p1u1d = ComputeAverage.computeAverageRobustness(ds2Times, ineffective_model, variables, pop.getFormulaNewGeneration(k + 1), vv);
                    p2u2d = ComputeAverage.computeAverageRobustness(ds2Times, normal_model, variables, pop.getFormulaNewGeneration(k + 1), vv);
//                p1u1d = smc(pop, pop.getFormulaNewGeneration(k + 1),ds2Times, ineffective_model);
//                p2u2d = smc(pop, pop.getFormulaNewGeneration(k + 1),ds2Times, normal_model);
                    double fitnessDescendant2 = fitness.compute(p1u1d[0], p2u2d[0], pop.getFormulaNewGeneration(k + 1).getFormulaSize(), p1u1d[1], p2u2d[1], 0);
                    if (fitnessDescendant2 > best) {
                        best = fitnessDescendant2;
                        bbbFormula = pop.getFormulaNewGeneration(k + 1);
                        bbbParamters = Arrays.copyOf(vv, vv.length);
                    }


                    if (fitnessDescendant1 < fitnessDescendant2) {
                        pop.removeFormulaNewPopulation(k);
                    } else {
                        pop.removeFormulaNewPopulation(k + 1);
                    }
                    //pop.mutateNewGeneration(k);
                    if (pop.getFormulaNewGeneration(k) != parent1 && pop.getFormulaNewGeneration(k) != parent2) {
                        vv = ComputeAverage.average(ds2Times, normal_model, ineffective_model, pop.getFormula(k), pop, new double[]{GeneticOptions.min_time_bound, GeneticOptions.max_time_bound});
                        //continuousOptimization(pop,variables, pop.getFormula(k), normal_model, ineffective_model,ds2Times);
                        p1u1[iterationIndex + 1][k] = ComputeAverage.computeAverageRobustness(ds2Times, ineffective_model, variables, pop.getFormula(k), vv);
                        p2u2[iterationIndex + 1][k] = ComputeAverage.computeAverageRobustness(ds2Times, normal_model, variables, pop.getFormula(k), vv);
//
                        // p1u1[iterationIndex + 1][k] = smc(pop, pop.getFormula(k), ds2Times,ineffective_model);
                        // p2u2[iterationIndex + 1][k] = smc(pop, pop.getFormula(k), ds2Times,normal_model);
                        fitnessScore[iterationIndex + 1][k] = fitness.compute(p1u1[iterationIndex + 1][k][0], p2u2[iterationIndex + 1][k][0], pop.getFormula(k + 1).getFormulaSize(), p1u1[iterationIndex][k][1], p2u2[iterationIndex][k][1], 0);
                        if (fitnessScore[iterationIndex + 1][k] > best) {
                            best = fitnessDescendant2;
                            bbbFormula = pop.getFormula(k);
                            bbbParamters = Arrays.copyOf(vv, vv.length);
                        }

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

                fitnessOrderedAscend = sortGetIndex(fitnessScore[iterationIndex + 1]);
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
//            if (migrationIndex <= migrationProbability) {
//                pop.removeFormulaNewPopulation(fitnessOrdered[fitnessOrdered.length - 1]);
//                int newIndividualIndex = pop.addRandomFormula();
//                continuousOptimization(pop,variables, pop.getFormula(newIndividualIndex), ineffective_model,normal_model, ds2Times);
//                p1u1[iterationIndex + 1][newIndividualIndex] = smc(pop, pop.getFormula(newIndividualIndex), ds2Times, ineffective_model_test);
//                p2u2[iterationIndex + 1][newIndividualIndex ] = smc(pop, pop.getFormula(newIndividualIndex), ds2Times, normal_model_test);
//                fitnessScore[iterationIndex + 1][newIndividualIndex] = fitness.compute(p1u1[iterationIndex + 1][newIndividualIndex][0], p2u2[iterationIndex + 1][newIndividualIndex][0], pop.getFormula(newIndividualIndex).getFormulaSize(), p1u1[iterationIndex][newIndividualIndex][1], p2u2[iterationIndex][newIndividualIndex][1], runs);
//                //fitnessOrderedAscend = sortGetIndex(fitnessScore[iterationIndex + 1]);
//                //fitnessOrdered = Arrays.stream(fitnessOrderedAscend).map(x -> -x).sorted().map(x -> -x).toArray();
//                fitnessOrderedAscend = sortGetIndex(fitnessScore[iterationIndex]);
//                fitnessOrdered = reverse(fitnessOrderedAscend);
//            }
                double bestNewFormulaFitness = fitnessScore[iterationIndex + 1][fitnessOrdered[1]];
                if (bestFormulaFitness == bestNewFormulaFitness) {
                    terminationIndex = terminationIndex + 1;
                } else {
                    terminationIndex = 0;
                    bestFormulaFitness = bestNewFormulaFitness;
                }
                // System.out.printf("TI: " + terminationIndex);
                for (int i = 0; i < pop.getPopulationSize(); i++) {
                    //   System.out.println(pop.getFormula(i));

                }

                for (int i = 0; i < pop.getNewPopulationSize(); i++) {
                    //   System.out.println(pop.getFormulaNewGeneration(i));

                }
                pop.finaliseNewGeneration();
                iterationIndex = iterationIndex + 1;
                //System.out.println();
            }
//        Formula bestFormula = pop.getFormula(fitnessOrdered[0]);
//
//
//
//        double p1bestFormula = p1u1[iterationIndex][fitnessOrdered[0]][0];
//        double p2bestFormula = p2u2[iterationIndex][fitnessOrdered[0]][0];
//        double u1bestFormula = p1u1[iterationIndex][fitnessOrdered[0]][1];
//        double u2bestFormula = p2u2[iterationIndex][fitnessOrdered[0]][1];
//        System.out.println("p1bestFormula= " + p1bestFormula);
//        System.out.println("p2bestFormula= " + p2bestFormula);
//        System.out.println("u1bestFormula= " + u1bestFormula);
//        System.out.println("u2bestFormula= " + u2bestFormula);

            //double[] vv = ComputeAverage.average(ds2Times,normal_model,ineffective_model, bestFormula, pop,new double[]{GeneticOptions.min_time_bound,GeneticOptions.max_time_bound});


            //double[] paramters = continuousOptimization(pop,variables, bestFormula, ineffective_model,normal_model, ds2Times);
            //pop.setParameters(bestFormula.getParameters(),vv);
//        System.out.println("ineffective= " + Arrays.toString(smcNew(ds2Times, ineffective_model_test, variables, bestFormula, vv)));
//        System.out.println("normal= " + Arrays.toString(smcNew(ds2Times, normal_model_test, variables, bestFormula, vv)));
//        System.out.println(bestFormula.toString());
//        System.out.println(Arrays.toString(vv));
            //System.out.println(Arrays.toString(paramters));
            formulae.add(bbbFormula);
            param.add(bbbParamters);
//            System.out.println("bbFormula:" + bbbFormula.toString());
//            System.out.println("bbPAram:" + Arrays.toString(bbbParamters));
//            System.out.println("ineffectiveB= " + Arrays.toString(smcNew(ds2Times, ineffective_model_test, variables, bbbFormula, bbbParamters)));
//            System.out.println("normalB= " + Arrays.toString(smcNew(ds2Times, normal_model_test, variables, bbbFormula, bbbParamters)));

        }
        for (int i = 0; i < formulae.size(); i++) {
            System.out.println("FFF");
           double[] vv = ComputeAverage.average(ds2Times, normal_model_pre, ineffective_model_pre,formulae.get(i), pop, new double[]{GeneticOptions.min_time_bound, GeneticOptions.max_time_bound});
            param.add(i,vv);
        }



        //double[] times, double[][] simulate, String[] variables, List<Formula> formula, List<double[]> param
        double[] previsionNormal=smcBoot(ds2Times,normal_model_test,variables,formulae,param);
        double[] previsionIneffective=smcBoot(ds2Times,ineffective_model_test,variables,formulae,param);
//        for (int i = 0; i < formulae.size(); i++) {
//            System.out.println(formulae.get(i).toString());
//            System.out.println(Arrays.toString(param.get(i)));
//        }

        for (Formula aFormulae : formulae) {
            System.out.println(aFormulae.toString());
        }
        System.out.println(Arrays.toString(previsionNormal));
        System.out.println(Arrays.toString(previsionIneffective));
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

    public static int[] reverse(int[] nums) {
        int[] reversed = new int[nums.length];
        for (int i = 0; i < nums.length; i++) {
            reversed[i] = nums[nums.length - 1 - i];
        }
        return reversed;
    }

    private static int[][] truncationSelection(int[] fitnessOrdered, int numParents, int i) {
        int[] candidateParents = Arrays.copyOfRange(fitnessOrdered, 0, i);
        int[][] parental = new int[numParents][];
        for (int j = 0; j < parental.length; j++) {
            parental[j] = new int[]{candidateParents[(int) Math.floor((Math.random() * candidateParents.length))], candidateParents[(int) Math.floor((Math.random() * candidateParents.length))]};
        }
        return parental;
    }

    static double[] smcNew(double[] times, double[][] simulate, String[] variables, Formula formula, double[] vv) {
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
        //builder.append("((G[Tl_1, Tu_1] (flow <= Theta_1) & G[Tl_2, Tu_2] (flow <= Theta_2) )  | F[Tl_3, Tu_3] (flow >= Theta_3))\n");
        // builder.append("((G[Tl_1, Tu_1] (flow <= Theta_1) & G[Tl_2, Tu_2] (flow <= Theta_2))  | F[Tl_3, Tu_3] (flow >= Theta_3))\n");
        MitlFactory factory = new MitlFactory(ns);
        String text = builder.toString();
        //System.out.println(text);
        MitlPropertiesList l = factory.constructProperties(text);
        MiTL prop = l.getProperties().get(0);

        for (int i = 0; i < simulate.length; i++) {
            Trajectory x = new Trajectory(times, ns, new double[][]{simulate[i]});
            b[i] = prop.evaluate(x, 0) ? 1 : 0;
        }
        double mean = Arrays.stream(b).sum() / b.length;
        double variance = Arrays.stream(b).map(x -> (x - mean) * (x - mean)).sum() / b.length;
        return new double[]{mean, variance};


    }

    static double[] smcBoot(double[] times, double[][] simulate, String[] variables, List<Formula> formula, List<double[]> param) {
        double[] b = new double[simulate.length];
        Context ns = new Context();
        for (String s : variables) {
            new Variable(s, ns);
        }

        for (int k = 0; k < simulate.length; k++) {
            int[] approx = new int[formula.size()];

            for (int i = 0; i < formula.size(); i++) {

                String[] parameters = formula.get(i).getParameters();
                StringBuilder builder = new StringBuilder();
                for (int j = 0; j < parameters.length; j++) {
                    builder.append("const double ").append(parameters[j]).append("=").append(param.get(i)[j]).append(";\n");
                }
                builder.append(formula.get(i).toString() + "\n");
                //builder.append("((G[Tl_1, Tu_1] (flow <= Theta_1) & G[Tl_2, Tu_2] (flow <= Theta_2) )  | F[Tl_3, Tu_3] (flow >= Theta_3))\n");
                // builder.append("((G[Tl_1, Tu_1] (flow <= Theta_1) & G[Tl_2, Tu_2] (flow <= Theta_2))  | F[Tl_3, Tu_3] (flow >= Theta_3))\n");
                MitlFactory factory = new MitlFactory(ns);
                String text = builder.toString();
                //System.out.println(text);
                MitlPropertiesList l = factory.constructProperties(text);
                MiTL prop = l.getProperties().get(0);


                Trajectory x = new Trajectory(times, ns, new double[][]{simulate[k]});
                approx[i] = prop.evaluate(x, 0) ? 1 : 0;
            }
            b[k]=Arrays.stream(approx).sum()>(double)approx.length/2.0? 1:0;

        }

            double mean = Arrays.stream(b).sum() / b.length;
            double variance = Arrays.stream(b).map(x -> (x - mean) * (x - mean)).sum() / b.length;




        return new double[]{mean, variance};


    }



}

