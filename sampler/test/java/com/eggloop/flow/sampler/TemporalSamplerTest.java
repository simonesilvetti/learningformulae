package com.eggloop.flow.sampler;

import org.junit.Test;

public class TemporalSamplerTest {

    @Test
    public void testSample() {
        TemporalSampler sampler = new TemporalSampler();
        double[][] sample = sampler.sample(10, new double[]{0, 1}, new double[]{1, 6});
        System.out.println();


    }
}