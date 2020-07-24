import React, { useState } from 'react';
import Alert from '../alerts/Alert'
import './upload.style.scss'
import {
    Button, TextField, Grid
} from '@material-ui/core'
import { makeStyles } from '@material-ui/core/styles';

const useStyles = makeStyles(theme => ({
    container: {
        minWidth: "100vh"
    },

    text: {
        width: '70%',
        textAlign: 'center',
    },

    submit: {
        height: '35px',
        fontFamily: 'Montserrat',
        textAlign: 'right',
    },

    item: {
        textAlign: 'center',
    },

    result: {
        textAlign: 'center',
        fontFamily: 'Montserrat',
    }
}));



const Upload = () => {
    const classes = useStyles();

    const [labelData, setLabelData] = useState({
        success: null,
        docs: 0,
        symbols: 0
    });

    const getAnnotatedDocument = () => {
        alert('Job Submitted!')



        fetch("http://localhost:5000/classify")
            .then(res => res.json())
            .then(data => {

                if (data.success) {
                    setLabelData((prevState) => ({
                        ...prevState,
                        success: data.success,
                        docs: data.docs,
                        symbols: data.symbols
                    }));
                    alert("Job successfully completed!")
                } else {
                    alert("Oops we encountered an error!")
                }

            })

    }

    return (
        <div>
            <Grid container spacing={3}
                direction="row"
                justify="center"
                alignItems="center">
                <Grid item xs={12} className={classes.item}>
                    <TextField id="outlined-basic" label="Enter local path..." variant="outlined" className={classes.text} margin='dense' autoFocus />
                </Grid>

                <Grid item xs={3} className={classes.item}>
                    <Button type="submit" variant="contained" color="primary" className={classes.submit} onClick={getAnnotatedDocument}>Submit</Button>
                </Grid>


                <Grid item xs={10} className={classes.result}>

                    {labelData.success === null ? (
                        <h2><i className="material-icons">loop</i>No Data to Display Yet</h2>
                    ) : <Alert docs={labelData.docs} symbols={labelData.symbols} />
                    }

                </Grid>

            </Grid>
        </div>
    )
};

export default Upload;