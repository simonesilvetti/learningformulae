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
import com.eggloop.flow.utils.data.TrajectoryMultiReconstruction;
import com.eggloop.flow.utils.files.Utils;

import java.util.*;
import java.util.stream.IntStream;

public class QUEST {
    public static void main(String[] args) {
        double start = System.currentTimeMillis();
        genetico();
        double finish = System.currentTimeMillis();
        System.out.println("TIME: "+(finish-start)/1000.0);
    }

    static double[] smcNew(double[] times, double[][][] simulate, String[] variables, Formula formula, double[] vv) {
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
            Trajectory x = new Trajectory(times, ns, simulate[i]);
            b[i] = prop.evaluate(x, 0) ? 1 : 0;
        }
        double mean = Arrays.stream(b).sum() / b.length;
        double variance = Arrays.stream(b).map(x -> (x - mean) * (x - mean)).sum() / b.length;
        return new double[]{mean, variance};


    }

    static double[] smcFormula(double[] times, double[][][] simulate, String[] variables, String formula) {
        double[] b = new double[simulate.length];
        Context ns = new Context();
        for (String s : variables) {
            new Variable(s, ns);
        }
        //String[] parameters = formula.getParameters();
        StringBuilder builder = new StringBuilder();

        //builder.append("((G[Tl_1, Tu_1] (flow <= Theta_1) & G[Tl_2, Tu_2] (flow <= Theta_2) )  | F[Tl_3, Tu_3] (flow >= Theta_3))\n");
        // builder.append("((G[Tl_1, Tu_1] (flow <= Theta_1) & G[Tl_2, Tu_2] (flow <= Theta_2))  | F[Tl_3, Tu_3] (flow >= Theta_3))\n");
        MitlFactory factory = new MitlFactory(ns);
        // String text = builder.toString();
        //System.out.println(text);
        MitlPropertiesList l = factory.constructProperties(formula);
        MiTL prop = l.getProperties().get(0);

        for (int i = 0; i < simulate.length; i++) {
            Trajectory x = new Trajectory(times, ns, simulate[i]);
            b[i] = prop.evaluate(x, 0) ? 1 : 0;
        }
        double mean = Arrays.stream(b).sum() / b.length;
        double variance = Arrays.stream(b).map(x -> (x - mean) * (x - mean)).sum() / b.length;
        return new double[]{mean, variance};


    }

    public static void genetico() {
        {
            Random ran = new Random(0);
            double[] ds2Labels = Utils.readVectorFromFile(TestReading.class.getClassLoader().getResource("data/calin/navalLabels").getPath());
            double[] ds2Times = Utils.readVectorFromFile(TestReading.class.getClassLoader().getResource("data/calin/navalTimes").getPath());
            double[][][] ds2SpatialValues = Utils.readMatrixMultiFromFile(ds2Times.length, TestReading.class.getClassLoader().getResource("data/calin/navalData").getPath());

            System.out.println("DONE");
            TrajectoryMultiReconstruction data = new TrajectoryMultiReconstruction(ds2Times, ds2SpatialValues, ds2Labels, 0.8, ran);
            data.split();
            double[][][] normal_model = data.getPositiveTrainingSet();
            double[][][] ineffective_model = data.getNegativeTrainingSet();
            double[][][] normal_model_test = data.getPoistiveTestSet();
            double[][][] ineffective_model_test = data.getNegativeTestSet();

            int N = 60;
            FormulaPopulation pop = new FormulaPopulation(N);
            String[] variables = new String[]{"x", "y"};
            double[] lower = new double[]{0, 0};
            double[] upper = new double[]{80, 45};
            for (int i = 0; i < variables.length; i++) {
                pop.addVariable(variables[i], lower[i], upper[i]);
            }

            GeneticOptions.setInit__prob_of_true_atom(0);
            GeneticOptions.setMin_time_bound(0);
            GeneticOptions.setMax_time_bound(300);



            List<Formula> rankFormulae = new ArrayList<>();
            List<double[]> rankParameters = new ArrayList<>();
            List<Double> rankScore = new ArrayList<>();

            //  pop.generateInitialPopulation();
            pop.addGeneticInitFormula(pop.getVariableNumber());
            for (int i = pop.getPopulationSize(); i <N ; i++) {
                pop.addRandomInitFormula();

            }

            for (int i = 0; i < pop.getPopulationSize(); i++) {
                double[] parameters = ComputeAverage.averageMulti(variables, ds2Times, normal_model, ineffective_model, pop.getFormula(i), pop, new double[]{GeneticOptions.min_time_bound, GeneticOptions.max_time_bound});
                double[] p1 = ComputeAverage.computeAverageRobustnessMulti(ds2Times, normal_model, variables, pop.getFormula(i), parameters);
                double[] p2 = ComputeAverage.computeAverageRobustnessMulti(ds2Times, ineffective_model, variables, pop.getFormula(i), parameters);
                double score = GeneticFitness.fitness(p1[0], p2[0], p1[1], p2[1]);
                rankFormulae.add(pop.getFormula(i));
                rankParameters.add(parameters);
                rankScore.add(score);
            }

            int[] sortedIndices = IntStream.range(0, rankScore.size())
                    .boxed().sorted(Comparator.comparingDouble(rankScore::get))
                    .mapToInt(ele -> ele).toArray();
            List<Formula> rankFormulaePOrd = new ArrayList<>();
            List<double[]> rankParametersPOrd = new ArrayList<>();
            List<Double> rankScorePOrd = new ArrayList<>();
            for (int sortedIndice : sortedIndices) {
                rankFormulaePOrd.add(rankFormulae.get(sortedIndice));
                rankParametersPOrd.add(rankParameters.get(sortedIndice));
                rankScorePOrd.add(rankScore.get(sortedIndice));
            }
            rankFormulae = rankFormulaePOrd;
            rankParameters = rankParametersPOrd;
            rankScore = rankScorePOrd;


            //SITUAZIONEINIZIALE


            pop.initialiseNewGeneration();

            //rankFormulae = rankFormulae.subList(rankScore.size() - rankScore.size() / 2, rankScore.size());
            //rankParameters = rankParameters.subList(rankScore.size() - rankScore.size() / 2, rankScore.size());
            //rankScore = rankScore.subList(rankScore.size() - rankScore.size() / 2, rankScore.size());









            for (int k = 0; k < 10; k++) {
                sortedIndices = IntStream.range(0, rankScore.size())
                        .boxed().sorted(Comparator.comparingDouble(rankScore::get))
                        .mapToInt(ele -> ele).toArray();
                rankFormulaePOrd = new ArrayList<>();
                rankParametersPOrd = new ArrayList<>();
                rankScorePOrd = new ArrayList<>();
                for (int sortedIndice : sortedIndices) {
                    rankFormulaePOrd.add(rankFormulae.get(sortedIndice));
                    rankParametersPOrd.add(rankParameters.get(sortedIndice));
                    rankScorePOrd.add(rankScore.get(sortedIndice));
                }
                rankFormulae = rankFormulaePOrd;
                rankParameters = rankParametersPOrd;
                rankScore = rankScorePOrd;


                //SITUAZIONEINIZIALE


                pop.initialiseNewGeneration();

                List<Formula> rankFormulaeParents = rankFormulae.subList(rankScore.size() - rankScore.size() / 2, rankScore.size());
                List<double[]> rankParametersParents = rankParameters.subList(rankScore.size() - rankScore.size() / 2, rankScore.size());
                List<Double> rankScoreParents = rankScore.subList(rankScore.size() - rankScore.size() / 2, rankScore.size());
                double[] cum = cum(rankScoreParents);
                for (int i = 0; i < rankFormulaeParents.size(); i++) {
                    //int a = ran.nextInt(rankFormulaeParents.size());
                    int a = extract( cum,ran);
                    int b = a;
                    while (b == a) {
                        // b = ran.nextInt(rankFormulaeParents.size());
                        b = extract( cum,ran);
                    }

                    int index1 = pop.addNewFormula(rankFormulaeParents.get(a));
                    int index2 = pop.addNewFormula(rankFormulaeParents.get(b));
                    //   if(ran.nextDouble()>0.8) {
                    pop.crossoverNewGeneration(index1, index2);
                    //   }
                    //  else{
                    //   pop.unionNewGeneration(index1,index2);
                    //   pop.mutateNewGeneration(index2);
                    // }
                }


                //CROSSOVER
                List<Formula> rankFormulaeNew = new ArrayList<>();
                List<double[]> rankParametersNew = new ArrayList<>();
                List<Double> rankScoreNew = new ArrayList<>();

                for (int i = 0; i < pop.getNewPopulationSize(); i++) {
                    double[] parameters = ComputeAverage.averageMulti(variables, ds2Times, normal_model, ineffective_model, pop.getFormulaNewGeneration(i), pop, new double[]{GeneticOptions.min_time_bound, GeneticOptions.max_time_bound});
                    try {
                        double[] p1 = ComputeAverage.computeAverageRobustnessMulti(ds2Times, normal_model, variables, pop.getFormulaNewGeneration(i), parameters);
                        double[] p2 = ComputeAverage.computeAverageRobustnessMulti(ds2Times, ineffective_model, variables, pop.getFormulaNewGeneration(i), parameters);
                        double score = GeneticFitness.fitness(p1[0], p2[0], p1[1], p2[1]);
                        rankFormulaeNew.add(pop.getFormulaNewGeneration(i));
                        rankParametersNew.add(parameters);
                        rankScoreNew.add(score);
                    }catch (Exception ex){
                        rankFormulaeNew.add(pop.getFormulaNewGeneration(i));
                        rankParametersNew.add(parameters);
                        rankScoreNew.add(0d);
                    }
                }
                //Ordino
                sortedIndices = IntStream.range(0, rankScoreNew.size())
                        .boxed().sorted(Comparator.comparingDouble(rankScoreNew::get))
                        .mapToInt(ele -> ele).toArray();
                rankFormulaePOrd = new ArrayList<>();
                rankParametersPOrd = new ArrayList<>();
                rankScorePOrd = new ArrayList<>();
                for (int sortedIndice : sortedIndices) {
                    rankFormulaePOrd.add(rankFormulaeNew.get(sortedIndice));
                    rankParametersPOrd.add(rankParametersNew.get(sortedIndice));
                    rankScorePOrd.add(rankScoreNew.get(sortedIndice));
                }
                List<Formula> rankFormulaeSon = rankFormulaePOrd.subList(rankScore.size() - rankScore.size() / 2, rankScore.size());
                List<double[]> rankParametersSon = rankParametersPOrd.subList(rankScore.size() - rankScore.size() / 2, rankScore.size());
                List<Double> rankScoreSon = rankScorePOrd.subList(rankScore.size() - rankScore.size() / 2, rankScore.size());
                rankFormulaeParents.addAll(rankFormulaeSon);
                rankParametersParents.addAll(rankParametersSon);
                rankScoreParents.addAll(rankScoreSon);
                rankFormulae = rankFormulaeParents;
                rankParameters = rankParametersParents;
                rankScore = rankScoreParents;
                System.out.println(rankScore);
            }
            sortedIndices = IntStream.range(0, rankScore.size())
                    .boxed().sorted(Comparator.comparingDouble(rankScore::get))
                    .mapToInt(ele -> ele).toArray();
            rankFormulaePOrd = new ArrayList<>();
            rankParametersPOrd = new ArrayList<>();
            rankScorePOrd = new ArrayList<>();
            for (int sortedIndice : sortedIndices) {
                rankFormulaePOrd.add(rankFormulae.get(sortedIndice));
                rankParametersPOrd.add(rankParameters.get(sortedIndice));
                rankScorePOrd.add(rankScore.get(sortedIndice));
            }
            rankFormulae = rankFormulaePOrd;
            rankParameters = rankParametersPOrd;
            rankScore = rankScorePOrd;

            double[] parameters = ComputeAverage.averageMultiSuper(variables, ds2Times, normal_model, ineffective_model, rankFormulae.get(rankFormulae.size()-1), pop, new double[]{GeneticOptions.min_time_bound, GeneticOptions.max_time_bound});





            System.out.println("__________________________________________________");


            System.out.println("bestFormula:" + rankFormulae.get(rankFormulae.size()-1).toString());
            // System.out.println("bestParam:" + Arrays.toString(parameters));
            System.out.println("bestParam:" + Arrays.toString(rankFormulae.get(rankFormulae.size() - 1).getParameters()) +":::::"+ Arrays.toString(rankParameters.get(rankFormulae.size()-1)));
            System.out.println("ineffectiveB= " + Arrays.toString(smcNew(ds2Times, ineffective_model_test, variables, rankFormulae.get(rankFormulae.size()-1), rankParameters.get(rankFormulae.size()-1))));
            System.out.println("normalB= " + Arrays.toString(smcNew(ds2Times, normal_model_test, variables, rankFormulae.get(rankFormulae.size()-1), rankParameters.get(rankFormulae.size()-1))));

//0.7//(((G_{[187,196)}x_{1}<19.8) \wedge (F_{[55.3,298)}x_{1}>40.8) ) \vee ((F_{[187,196)}x_{1}>19.8) \wedge ((G_{[94.9,296)}x_{2}<32.2)  \vee ((F_{[94.9,296)}x_{2}>32.2) \wedge (((G_{[50.2,274)}x_{2}>29.6) \wedge (G_{[125,222)}x_{1}<47) ) \vee ((F_{[50.2,274)}x_{2}<29.6) \wedge (G_{[206,233)}x_{1}<16.7) ))))))
//0.5//(((G_{[33.3,194)}x_{1}>19.5) \wedge ((G_{[88.5,299)}x_{2}<32)  \vee ((F_{[88.5,299)}x_{2}>32) \wedge (((G_{[142,296)}x_{2}>29.6) \wedge (G_{[80.1,251)}x_{1}<52.1) ) \vee ((F_{[142,296)}x_{2}<29.6) \wedge (G_{[103,250)}x_{2}<29.4) ))))) \vee ((F_{[33.3,194)}x_{1}<19.5) \wedge (F_{[59.6,84.2)}x_{1}>41.8) ))
//String calin = "(((G[187,196]flow<19.8) & (F[55.3,298]flow>40.8) ) | ((F[187,196]]flow>19.8) & ((G[94.9,296]flow1<32.2)  | ((F[94.9,296]flow1>32.2) & (((G[50.2,274]flow1>29.6) & (G[125,222]flow<47) ) | ((F[50.2,274]flow1<29.6) & (G[206,233]flow<16.7) ))))))"

//            System.out.println("::::::::::::::::::::::::::::::::::::::::::::::::::::::::::");
//            System.out.println("a");
//            for (int i = rankFormulae.size()/2; i < rankFormulae.size(); i++) {
//                System.out.println(rankFormulae.get(i).toString());
//                double[] vv = ComputeAverage.averageMulti(variables, ds2Times, normal_model, ineffective_model, rankFormulae.get(i), pop, new double[]{GeneticOptions.min_time_bound, GeneticOptions.max_time_bound});
//                rankParameters.set(i,vv);
//                System.out.println(Arrays.toString(rankParameters.get(i)));
//            }
//
//            System.out.println("ineffectiveB= " + Arrays.toString(smcBoot(ds2Times, ineffective_model_test, variables, rankFormulae.subList(rankParameters.size()/2,rankParameters.size()), rankParameters.subList(rankParameters.size()/2,rankParameters.size()))));
//            System.out.println("normalB= " + Arrays.toString(smcBoot(ds2Times, normal_model_test, variables, rankFormulae.subList(rankParameters.size()/2,rankParameters.size()), rankParameters.subList(rankParameters.size()/2,rankParameters.size()))));


        }

    }

    public static void geneticoProb() {
        {
            Random ran = new Random(0);
            double[] ds2Labels = Utils.readVectorFromFile(TestReading.class.getClassLoader().getResource("data/calin/navalLabels").getPath());
            double[] ds2Times = Utils.readVectorFromFile(TestReading.class.getClassLoader().getResource("data/calin/navalTimes").getPath());
            double[][][] ds2SpatialValues = Utils.readMatrixMultiFromFile(ds2Times.length, TestReading.class.getClassLoader().getResource("data/calin/navalData").getPath());
            System.out.println("DONE");
            TrajectoryMultiReconstruction data = new TrajectoryMultiReconstruction(ds2Times, ds2SpatialValues, ds2Labels, 0.6, ran);
            data.split();
            double[][][] normal_model = data.getPositiveTrainingSet();
            double[][][] ineffective_model = data.getNegativeTrainingSet();
            double[][][] normal_model_test = data.getPoistiveTestSet();
            double[][][] ineffective_model_test = data.getNegativeTestSet();
            int N = 80;

            double crossoverProbability = 1;
            double migrationProbability = 0.0;
            int numParents = N;
            int termination = 4;
            double terminationIndex = 0;
            int iterationIndex = 0;
            double best = -Double.POSITIVE_INFINITY;
            Formula bbbFormula = null;
            double[] bbbParamters = null;
            FormulaPopulation pop = new FormulaPopulation(N);
            String[] variables = new String[]{"flow", "flow1"};
            double[] lower = new double[]{0, 0};
            double[] upper = new double[]{80, 45};
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
            GeneticOptions.setMax_time_bound(300);



            List<Formula> rankFormulae = new ArrayList<>();
            List<double[]> rankParameters = new ArrayList<>();
            List<Double> rankScore = new ArrayList<>();

            pop.generateInitialPopulation();

            for (int i = 0; i < pop.getPopulationSize(); i++) {
                double[] parameters = ComputeAverage.probMulti(variables, ds2Times, normal_model, ineffective_model, pop.getFormula(i), pop, new double[]{GeneticOptions.min_time_bound, GeneticOptions.max_time_bound});
                try {
                    double[] p1 = ComputeAverage.computeProbMulti(ds2Times, normal_model, variables, pop.getFormula(i), parameters);
                    double[] p2 = ComputeAverage.computeProbMulti(ds2Times, ineffective_model, variables, pop.getFormula(i), parameters);
                    double score = GeneticFitness.fitnessProb(p1[0], p2[0], p1[1], p2[1]);
                    rankFormulae.add(pop.getFormula(i));
                    rankParameters.add(parameters);
                    rankScore.add(score);
                }catch (Exception ex){
                    rankFormulae.add(pop.getFormula(i));
                    rankParameters.add(parameters);
                    rankScore.add(0d);
                }
            }

            int[] sortedIndices = IntStream.range(0, rankScore.size())
                    .boxed().sorted(Comparator.comparingDouble(rankScore::get))
                    .mapToInt(ele -> ele).toArray();
            List<Formula> rankFormulaePOrd = new ArrayList<>();
            List<double[]> rankParametersPOrd = new ArrayList<>();
            List<Double> rankScorePOrd = new ArrayList<>();
            for (int sortedIndice : sortedIndices) {
                rankFormulaePOrd.add(rankFormulae.get(sortedIndice));
                rankParametersPOrd.add(rankParameters.get(sortedIndice));
                rankScorePOrd.add(rankScore.get(sortedIndice));
            }
            rankFormulae = rankFormulaePOrd;
            rankParameters = rankParametersPOrd;
            rankScore = rankScorePOrd;


            //SITUAZIONEINIZIALE


            pop.initialiseNewGeneration();

            rankFormulae = rankFormulae.subList(rankScore.size() - rankScore.size() / 4, rankScore.size());
            rankParameters = rankParameters.subList(rankScore.size() - rankScore.size() / 4, rankScore.size());
            rankScore = rankScore.subList(rankScore.size() - rankScore.size() / 4, rankScore.size());









            for (int k = 0; k < 12; k++) {
                sortedIndices = IntStream.range(0, rankScore.size())
                        .boxed().sorted(Comparator.comparingDouble(rankScore::get))
                        .mapToInt(ele -> ele).toArray();
                rankFormulaePOrd = new ArrayList<>();
                rankParametersPOrd = new ArrayList<>();
                rankScorePOrd = new ArrayList<>();
                for (int sortedIndice : sortedIndices) {
                    rankFormulaePOrd.add(rankFormulae.get(sortedIndice));
                    rankParametersPOrd.add(rankParameters.get(sortedIndice));
                    rankScorePOrd.add(rankScore.get(sortedIndice));
                }
                rankFormulae = rankFormulaePOrd;
                rankParameters = rankParametersPOrd;
                rankScore = rankScorePOrd;


                //SITUAZIONEINIZIALE


                pop.initialiseNewGeneration();

                List<Formula> rankFormulaeParents = rankFormulae.subList(rankScore.size() - rankScore.size() / 2, rankScore.size());
                List<double[]> rankParametersParents = rankParameters.subList(rankScore.size() - rankScore.size() / 2, rankScore.size());
                List<Double> rankScoreParents = rankScore.subList(rankScore.size() - rankScore.size() / 2, rankScore.size());
                for (int i = 0; i < rankFormulaeParents.size(); i++) {
                    int a = ran.nextInt(rankFormulaeParents.size());
                    int b = a;
                    while (b == a) {
                        b = ran.nextInt(rankFormulaeParents.size());
                    }
                    int index1 = pop.addNewFormula(rankFormulaeParents.get(a));
                    int index2 = pop.addNewFormula(rankFormulaeParents.get(b));
                    pop.crossoverNewGeneration(index1, index2);
                }


                //CROSSOVER
                List<Formula> rankFormulaeNew = new ArrayList<>();
                List<double[]> rankParametersNew = new ArrayList<>();
                List<Double> rankScoreNew = new ArrayList<>();

                for (int i = 0; i < pop.getNewPopulationSize(); i++) {
                    double[] parameters = ComputeAverage.probMulti(variables, ds2Times, normal_model, ineffective_model, pop.getFormulaNewGeneration(i), pop, new double[]{GeneticOptions.min_time_bound, GeneticOptions.max_time_bound});
                    try {
                        double[] p1 = ComputeAverage.computeProbMulti(ds2Times, normal_model, variables, pop.getFormulaNewGeneration(i), parameters);
                        double[] p2 = ComputeAverage.computeProbMulti(ds2Times, ineffective_model, variables, pop.getFormulaNewGeneration(i), parameters);
                        double score = GeneticFitness.fitnessProb(p1[0], p2[0], p1[1], p2[1]);
                        rankFormulaeNew.add(pop.getFormulaNewGeneration(i));
                        rankParametersNew.add(parameters);
                        rankScoreNew.add(score);
                    }catch (Exception ex){
                        rankFormulaeNew.add(pop.getFormulaNewGeneration(i));
                        rankParametersNew.add(parameters);
                        rankScoreNew.add(0d);
                    }
                }
                //Ordino
                sortedIndices = IntStream.range(0, rankScoreNew.size())
                        .boxed().sorted(Comparator.comparingDouble(rankScoreNew::get))
                        .mapToInt(ele -> ele).toArray();
                rankFormulaePOrd = new ArrayList<>();
                rankParametersPOrd = new ArrayList<>();
                rankScorePOrd = new ArrayList<>();
                for (int sortedIndice : sortedIndices) {
                    rankFormulaePOrd.add(rankFormulaeNew.get(sortedIndice));
                    rankParametersPOrd.add(rankParametersNew.get(sortedIndice));
                    rankScorePOrd.add(rankScoreNew.get(sortedIndice));
                }
                List<Formula> rankFormulaeSon = rankFormulaePOrd.subList(rankScore.size() - rankScore.size() / 2, rankScore.size());
                List<double[]> rankParametersSon = rankParametersPOrd.subList(rankScore.size() - rankScore.size() / 2, rankScore.size());
                List<Double> rankScoreSon = rankScorePOrd.subList(rankScore.size() - rankScore.size() / 2, rankScore.size());
                rankFormulaeParents.addAll(rankFormulaeSon);
                rankParametersParents.addAll(rankParametersSon);
                rankScoreParents.addAll(rankScoreSon);
                rankFormulae = rankFormulaeParents;
                rankParameters = rankParametersParents;
                rankScore = rankScoreParents;
                System.out.println(rankScore);
            }
            sortedIndices = IntStream.range(0, rankScore.size())
                    .boxed().sorted(Comparator.comparingDouble(rankScore::get))
                    .mapToInt(ele -> ele).toArray();
            rankFormulaePOrd = new ArrayList<>();
            rankParametersPOrd = new ArrayList<>();
            rankScorePOrd = new ArrayList<>();
            for (int sortedIndice : sortedIndices) {
                rankFormulaePOrd.add(rankFormulae.get(sortedIndice));
                rankParametersPOrd.add(rankParameters.get(sortedIndice));
                rankScorePOrd.add(rankScore.get(sortedIndice));
            }
            rankFormulae = rankFormulaePOrd;
            rankParameters = rankParametersPOrd;
            rankScore = rankScorePOrd;

            double[] parameters = ComputeAverage.probMultiSuper(variables, ds2Times, normal_model, ineffective_model, rankFormulae.get(rankFormulae.size()-1), pop, new double[]{GeneticOptions.min_time_bound, GeneticOptions.max_time_bound});





            System.out.println("__________________________________________________");


            System.out.println("bestFormula:" + rankFormulae.get(rankFormulae.size()-1).toString());
            System.out.println("bestParam:" + Arrays.toString(parameters));
            System.out.println("ineffectiveB= " + Arrays.toString(smcNew(ds2Times, ineffective_model_test, variables, rankFormulae.get(rankFormulae.size()-1), parameters)));
            System.out.println("normalB= " + Arrays.toString(smcNew(ds2Times, normal_model_test, variables, rankFormulae.get(rankFormulae.size()-1), parameters)));

//0.7//(((G_{[187,196)}x_{1}<19.8) \wedge (F_{[55.3,298)}x_{1}>40.8) ) \vee ((F_{[187,196)}x_{1}>19.8) \wedge ((G_{[94.9,296)}x_{2}<32.2)  \vee ((F_{[94.9,296)}x_{2}>32.2) \wedge (((G_{[50.2,274)}x_{2}>29.6) \wedge (G_{[125,222)}x_{1}<47) ) \vee ((F_{[50.2,274)}x_{2}<29.6) \wedge (G_{[206,233)}x_{1}<16.7) ))))))
//0.5//(((G_{[33.3,194)}x_{1}>19.5) \wedge ((G_{[88.5,299)}x_{2}<32)  \vee ((F_{[88.5,299)}x_{2}>32) \wedge (((G_{[142,296)}x_{2}>29.6) \wedge (G_{[80.1,251)}x_{1}<52.1) ) \vee ((F_{[142,296)}x_{2}<29.6) \wedge (G_{[103,250)}x_{2}<29.4) ))))) \vee ((F_{[33.3,194)}x_{1}<19.5) \wedge (F_{[59.6,84.2)}x_{1}>41.8) ))
//String calin = "(((G[187,196]flow<19.8) & (F[55.3,298]flow>40.8) ) | ((F[187,196]]flow>19.8) & ((G[94.9,296]flow1<32.2)  | ((F[94.9,296]flow1>32.2) & (((G[50.2,274]flow1>29.6) & (G[125,222]flow<47) ) | ((F[50.2,274]flow1<29.6) & (G[206,233]flow<16.7) ))))))"

            System.out.println("::::::::::::::::::::::::::::::::::::::::::::::::::::::::::");
            System.out.println("a");
            for (int i = rankFormulae.size()/2; i < rankFormulae.size(); i++) {
                System.out.println(rankFormulae.get(i).toString());
            }

            System.out.println("ineffectiveB= " + Arrays.toString(smcBoot(ds2Times, ineffective_model_test, variables, rankFormulae.subList(rankParameters.size()/2,rankParameters.size()), rankParameters.subList(rankParameters.size()/2,rankParameters.size()))));
            System.out.println("normalB= " + Arrays.toString(smcBoot(ds2Times, normal_model_test, variables, rankFormulae.subList(rankParameters.size()/2,rankParameters.size()), rankParameters.subList(rankParameters.size()/2,rankParameters.size()))));


        }

    }


    static double[] smcBoot(double[] times, double[][][] simulate, String[] variables, List<Formula> formula, List<double[]> param) {
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


                Trajectory x = new Trajectory(times, ns, simulate[k]);
                approx[i] = prop.evaluate(x, 0) ? 1 : 0;
            }
            b[k]=Arrays.stream(approx).sum()>(double)approx.length/2.0? 1:0;

        }

        double mean = Arrays.stream(b).sum() / b.length;
        double variance = Arrays.stream(b).map(x -> (x - mean) * (x - mean)).sum() / b.length;




        return new double[]{mean, variance};


    }

    public static  double[]  prova(double[] times,double[][][] simulate,String[] variables){
        String formula = "G[87, 152] ((x >= 24 & y <= 36))\n";

        double[] b = new double[simulate.length];
        Context ns = new Context();
        for (String s : variables) {
            new Variable(s, ns);
        }

        //builder.append("((G[Tl_1, Tu_1] (flow <= Theta_1) & G[Tl_2, Tu_2] (flow <= Theta_2) )  | F[Tl_3, Tu_3] (flow >= Theta_3))\n");
        // builder.append("((G[Tl_1, Tu_1] (flow <= Theta_1) & G[Tl_2, Tu_2] (flow <= Theta_2))  | F[Tl_3, Tu_3] (flow >= Theta_3))\n");
        MitlFactory factory = new MitlFactory(ns);
        //System.out.println(text);
        MitlPropertiesList l = factory.constructProperties(formula);
        MiTL prop = l.getProperties().get(0);

        for (int i = 0; i < simulate.length; i++) {
            Trajectory x = new Trajectory(times, ns, simulate[i]);
            b[i] = prop.evaluate(x, 0) ? 1 : 0;
        }
        double mean = Arrays.stream(b).sum() / b.length;
        double variance = Arrays.stream(b).map(x -> (x - mean) * (x - mean)).sum() / b.length;
        return new double[]{mean, variance};




    }


    public static double[] cum(List<Double> w){
        double[] res = new double[w.size()+1];
        for (int i = 1; i < res.length; i++) {
            res[i]=res[i-1]+w.get(i-1);
        }
        return Arrays.stream(res).map(s->s/res[res.length-1]).toArray();
    }

    public static int extract(double[] cum, Random ran){
        double r = ran.nextDouble();
        for (int i = 0; i < cum.length; i++) {
            if(cum[i]>r){
                return i-1;
            }
        }
        return cum.length-1;
    }
}
