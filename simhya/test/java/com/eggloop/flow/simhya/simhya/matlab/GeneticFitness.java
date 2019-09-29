package com.eggloop.flow.simhya.simhya.matlab;

public class GeneticFitness {

    public static double fitness(double p1, double p2, double v1, double v2){
        return (p1-p2)/(3*Math.abs(v1+v2));
    }
    public static double fitnessProb(double p1, double p2, double v1, double v2){
        return Math.max(Math.log((p1+1)/(p2+1)),Math.log((p2+1)/(p1+1)));
    }

}
