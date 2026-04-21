#![allow(non_snake_case)]
//! Rust port of `apps/flatter.cpp`. Command-line entry point.
//!
//! Flags mirror the C++ version exactly (see `print_help` below) so shell
//! pipelines built against the original binary keep working with this one.

use std::fs::File;
use std::io::{self, BufReader, Read, Write};
use std::process::ExitCode;
use std::time::Instant;

use flatter_rs::lattice::Lattice;
use flatter_rs::reduction::{reduce, LatticeReductionGoal, LatticeReductionParams};

fn print_help() {
    eprintln!("Usage: flatter [-h] [-v] [-alpha ALPHA | -rhf RHF | -delta DELTA] [-logcond LOGCOND] [INFILE [OUTFILE]]");
    eprintln!("\tINFILE -\tinput lattice (FPLLL format). Defaults to STDIN");
    eprintln!("\tOUTFILE -\toutput lattice (FPLLL format). Defaults to STDOUT");
    eprintln!("\t-h -\thelp message.");
    eprintln!("\t-v -\tverbose output.");
    eprintln!("\t-q -\tdo not output lattice.");
    eprintln!("\t-p -\toutput profiles.");
    eprintln!("\tReduction quality - up to one of the following. Default to RHF 1.0219");
    eprintln!("\t\t-alpha ALPHA -\tReduce to given parameter alpha");
    eprintln!("\t\t-rhf RHF -\tReduce analogous to given root hermite factor");
    eprintln!("\t\t-delta DELTA -\tReduce analogous to LLL with particular delta (approximate)");
    eprintln!("\t-logcond LOGCOND -\tBound on condition number.");
}

#[derive(Default)]
struct Args {
    show_help: bool,
    verbose: bool,
    quiet: bool,
    show_profile: bool,
    red_quality_set: bool,
    logcond_set: bool,
    alpha: f64,
    logcond: f64,
    input_path: Option<String>,
    output_path: Option<String>,
}

fn parse_args<I: IntoIterator<Item = String>>(argv: I) -> Result<Args, String> {
    let mut args = Args::default();
    let mut it = argv.into_iter().skip(1);
    while let Some(a) = it.next() {
        match a.as_str() {
            "-h" => {
                args.show_help = true;
                return Ok(args);
            }
            "-v" => args.verbose = true,
            "-q" => {
                if args.output_path.is_some() {
                    return Err("Cannot set output file in quiet (-q) mode".into());
                }
                args.quiet = true;
            }
            "-p" => args.show_profile = true,
            "-alpha" | "-rhf" | "-delta" => {
                if args.red_quality_set {
                    return Err(format!("repeated reduction-quality flag: {}", a));
                }
                let v: f64 = it
                    .next()
                    .ok_or_else(|| format!("missing value for {}", a))?
                    .parse()
                    .map_err(|_| format!("bad value for {}", a))?;
                args.alpha = match a.as_str() {
                    "-alpha" => v,
                    "-rhf" => 2.0 * v.log2(),
                    "-delta" => (0.255 / v).powi(2),
                    _ => unreachable!(),
                };
                args.red_quality_set = true;
            }
            "-logcond" => {
                if args.logcond_set {
                    return Err("repeated -logcond".into());
                }
                let v: f64 = it
                    .next()
                    .ok_or("missing value for -logcond")?
                    .parse()
                    .map_err(|_| "bad value for -logcond".to_string())?;
                args.logcond = v;
                args.logcond_set = true;
            }
            _ if args.input_path.is_none() => {
                args.input_path = Some(a);
            }
            _ if args.output_path.is_none() => {
                if args.quiet {
                    return Err("Cannot set output file in quiet (-q) mode".into());
                }
                args.output_path = Some(a);
            }
            _ => return Err("Too many input/output files specified".into()),
        }
    }
    if !args.red_quality_set {
        // Matches apps/flatter.cpp:148 — RHF 1.0219 in alpha form.
        args.alpha = 0.06250805094100162;
    }
    Ok(args)
}

fn read_input(path: Option<&String>) -> io::Result<Lattice> {
    let boxed: Box<dyn Read> = match path {
        Some(p) if p != "-" => Box::new(File::open(p)?),
        _ => Box::new(io::stdin()),
    };
    let mut r = BufReader::new(boxed);
    Lattice::read_fplll(&mut r)
}

fn open_output(path: Option<&String>) -> io::Result<Box<dyn Write>> {
    match path {
        Some(p) if p != "-" => Ok(Box::new(File::create(p)?)),
        _ => Ok(Box::new(io::stdout())),
    }
}

fn mpz_bits_max(L: &Lattice) -> u64 {
    let mut mx: u64 = 0;
    for e in &L.basis.data {
        let b = e.significant_bits() as u64;
        if b > mx {
            mx = b;
        }
    }
    mx
}

fn run() -> Result<(), String> {
    let args = parse_args(std::env::args().collect::<Vec<_>>())
        .map_err(|e| e)?;
    if args.show_help {
        print_help();
        return Ok(());
    }

    let mut L = read_input(args.input_path.as_ref())
        .map_err(|e| format!("reading input: {}", e))?;

    if args.verbose {
        eprintln!(
            "Input lattice of rank {} and dimension {}",
            L.rank,
            L.dimension()
        );
        eprintln!(
            "Largest entry is {} bits in length.",
            mpz_bits_max(&L)
        );
        if L.basis.is_upper_triangular() {
            // Emit input profile from the diagonal (log2 |B_ii|).
            if args.show_profile {
                eprintln!("Input profile:");
                let mut first = true;
                for i in 0..L.rank {
                    let d = L.basis.get(i, i).to_f64().abs();
                    let v = if d > 0.0 { d.log2() } else { f64::NEG_INFINITY };
                    if !first {
                        eprint!(" ");
                    }
                    eprint!("{}", v);
                    first = false;
                }
                eprintln!();
            }
        } else {
            eprintln!("Skipped determining input profile, as input is not lower-triangular.");
        }
        eprintln!(
            "Target reduction quality alpha = {}, rhf = {}",
            args.alpha,
            (2f64).powf(args.alpha / 2.0)
        );
    }

    let goal = LatticeReductionGoal::from_slope(L.rank, args.alpha);
    let mut params = LatticeReductionParams::from_goal(goal);
    if args.logcond_set {
        params.log_cond = args.logcond;
    }

    let t0 = Instant::now();
    let report = reduce(&mut L, &params);
    let elapsed = t0.elapsed();
    L.update_rank();

    if args.verbose {
        eprintln!(
            "Reduction took {} milliseconds. (algorithm={}, iters={})",
            elapsed.as_millis(),
            report.algorithm,
            report.iterations
        );
    }

    if args.show_profile {
        eprintln!("Output profile:");
        let mut first = true;
        for i in 0..L.rank {
            if !first {
                eprint!(" ");
            }
            eprint!("{}", L.profile[i]);
            first = false;
        }
        eprintln!();
    }

    if args.verbose {
        let n = L.rank.max(1);
        let logdet: f64 = L.profile.logdet();
        let log_rhf = (L.profile[0] - logdet / n as f64) / n as f64;
        let rhf = (2f64).powf(log_rhf);
        let drop = L.profile.get_drop();
        let alpha = drop / n as f64;
        eprintln!(
            "Achieved reduction quality alpha = {}, rhf = {}",
            alpha, rhf
        );
    }

    if !args.quiet {
        let mut out = open_output(args.output_path.as_ref())
            .map_err(|e| format!("opening output: {}", e))?;
        write!(out, "{}", L).map_err(|e| format!("writing output: {}", e))?;
    }

    Ok(())
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("{}", e);
            print_help();
            ExitCode::FAILURE
        }
    }
}
