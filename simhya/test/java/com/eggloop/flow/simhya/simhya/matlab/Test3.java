package com.eggloop.flow.simhya.simhya.matlab;

import com.eggloop.flow.numeric.optimization.ObjectiveFunction;
import com.eggloop.flow.numeric.optimization.PointValue;
import com.eggloop.flow.numeric.optimization.methods.ConjugateGradientApache;
import com.eggloop.flow.numeric.optimization.methods.PowellMethodApacheConstrained;
import org.apache.commons.math.analysis.MultivariateRealFunction;
import org.apache.commons.math.optimization.GoalType;
import org.apache.commons.math.optimization.RealPointValuePair;
import org.apache.commons.math.optimization.direct.NelderMead;
import org.apache.commons.math.optimization.linear.SimplexSolver;
import org.apache.commons.math3.analysis.MultivariateFunction;
import org.apache.commons.math3.linear.ConjugateGradient;
import org.apache.commons.math3.optim.InitialGuess;
import org.apache.commons.math3.optim.MaxEval;
import org.apache.commons.math3.optim.OptimizationData;
import org.apache.commons.math3.optim.PointValuePair;
import org.apache.commons.math3.optim.nonlinear.scalar.noderiv.NelderMeadSimplex;
import org.apache.commons.math3.optim.nonlinear.scalar.noderiv.SimplexOptimizer;
import org.apache.commons.math3.util.FastMath;

import java.util.Arrays;

public class Test3 {

    public static void main(String[] args) {
        PowellMethodApacheConstrained alg = new PowellMethodApacheConstrained();
        //ConjugateGradientApache alg = new ConjugateGradientApache();
        //NelderMead alg = new NelderMead();
        ObjectiveFunction function = new Funzione();
        //PowellOptimizer optimizer = new PowellOptimizer();
        double[] start = new double[]{0.5,0.5};
        PointValue best = alg.optimise(function, start,new double[]{0,0},new double[]{1,1});
        //PointValue best = alg.optimise(function, start);
        //PointValue best = alg.optimise(function, start);
        double[] bestPar = best.getPoint();
        System.out.println(Arrays.toString(bestPar));



    }
}
class Funzione implements ObjectiveFunction {
    @Override
    public double getValueAt(double... point) {
        System.out.println(Arrays.toString(point));
        return  point[0]* point[0] +point[1];
       // return 1;
    }




}
class FourExtrema implements MultivariateFunction,OptimizationData
{
    // The following function has 4 local extrema.
    final double xM = -3.841947088256863675365;
    final double yM = -1.391745200270734924416;
    final double xP = 0.2286682237349059125691;
    final double yP = -yM;
    final double valueXmYm = 0.2373295333134216789769; // Local maximum.
    final double valueXmYp = -valueXmYm; // Local minimum.
    final double valueXpYm = -0.7290400707055187115322; // Global minimum.
    final double valueXpYp = -valueXpYm; // Global maximum.

    public double value(double[] variables)
    {
        final double x = variables[0];
        final double y = variables[1];
        return (x == 0 || y == 0) ? 0 : FastMath.atan(x)
                * FastMath.atan(x + 2) * FastMath.atan(y) * FastMath.atan(y)
                / (x * y);
    }
}
